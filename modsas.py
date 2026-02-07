import streamlit as st
import io
import requests
import re
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from Bio import SeqIO
from io import StringIO

# ==========================================
# 1. TRY TO IMPORT ALL REQUIRED LIBRARIES
# ==========================================
try:
    # Biopython Imports
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction, molecular_weight, seq3
    from Bio.SeqUtils.MeltingTemp import Tm_NN
    from Bio.Align import PairwiseAligner
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.Data import CodonTable
    
    # Visualization imports
    import py3Dmol
    from stmol import showmol
    
    # Additional imports
    from collections import Counter
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    
    LIBRARIES_OK = True
except ImportError as e:
    st.error(f"Required library missing: {e}")
    st.info("Please install required packages: pip install biopython reportlab matplotlib py3dmol stmol requests plotly pandas numpy")
    LIBRARIES_OK = False

# ==========================================
# 2. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Generize SaaS | Bioinformatics Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #2e86de;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0abde3;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #dcdcdc;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff3333;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CORE LOGIC UTILITIES (Enhanced)
# ==========================================

def clean_sequence(seq):
    """Removes whitespace and numbers from sequence."""
    return re.sub(r'[\s\d]', '', seq).upper()

def validate_dna_sequence(sequence):
    """Validates DNA sequence."""
    if not sequence:
        return False, "Sequence is empty"
    
    valid_chars = set("ATCGN")
    seq_chars = set(sequence.upper())
    
    if not seq_chars.issubset(valid_chars):
        invalid_chars = seq_chars - valid_chars
        return False, f"Invalid characters found: {''.join(invalid_chars)}"
    
    return True, "Valid DNA sequence"

def calculate_dna_metrics(sequence):
    """Calculates GC, Tm, and generates protein translation."""
    try:
        seq_obj = Seq(sequence)
        gc = gc_fraction(seq_obj) * 100
        
        # Tm calculation requires checking for non-ATCG characters
        if set(sequence).issubset({"A", "T", "C", "G"}):
            tm = Tm_NN(seq_obj)
        else:
            tm = 0.0
        
        protein = str(seq_obj.translate(to_stop=False))
        return gc, tm, protein, None
    except Exception as e:
        return 0, 0, "", f"Error in calculation: {str(e)}"

def find_orfs(sequence, min_orf_length=30):
    """Find all ORFs in a DNA sequence from start to stop codons."""
    seq = Seq(sequence)
    orfs = []
    start_codons = ['ATG']
    stop_codons = ['TAA', 'TAG', 'TGA']
    
    # Check all three reading frames
    for frame in range(3):
        trans_seq = seq[frame:]
        protein = trans_seq.translate(to_stop=False)
        
        current_position = 0
        while current_position < len(protein):
            # Find next M (start)
            start_index = protein.find('M', current_position)
            if start_index == -1:
                break
                
            # Find next stop after this start
            stop_index = protein.find('*', start_index)
            if stop_index == -1:
                break
                
            # Calculate ORF length in nucleotides
            orf_nuc_length = (stop_index - start_index + 1) * 3
            
            if orf_nuc_length >= min_orf_length:
                # Calculate positions in original DNA sequence
                dna_start = frame + start_index * 3
                dna_end = frame + stop_index * 3 + 3
                
                orf_protein = protein[start_index:stop_index]
                
                orfs.append({
                    'frame': frame + 1,
                    'start': dna_start + 1,  # Convert to 1-based indexing
                    'end': dna_end,
                    'length': orf_nuc_length,
                    'protein': str(orf_protein),
                    'protein_length': len(orf_protein)
                })
            
            current_position = stop_index + 1
    
    # Sort ORFs by length (longest first)
    orfs.sort(key=lambda x: x['length'], reverse=True)
    return orfs

def calculate_coding_ratio(sequence, orfs):
    """Calculate coding ratio (coding nucleotides / total nucleotides)."""
    if not orfs:
        return 0.0
    
    total_length = len(sequence)
    coding_length = 0
    
    # Merge overlapping ORFs to avoid double counting
    intervals = []
    for orf in orfs:
        intervals.append((orf['start'] - 1, orf['end'] - 1))  # Convert to 0-based
    
    # Sort intervals by start position
    intervals.sort(key=lambda x: x[0])
    
    # Merge intervals
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    # Calculate total coding length
    for start, end in merged:
        coding_length += (end - start + 1)
    
    return (coding_length / total_length) * 100 if total_length > 0 else 0

def analyze_amino_acids(protein_sequence):
    """Analyze amino acid composition of a protein sequence."""
    try:
        if not protein_sequence:
            return {}
        
        # Use BioPython's ProteinAnalysis
        prot_analysis = ProteinAnalysis(protein_sequence.replace('*', ''))
        
        amino_acid_composition = prot_analysis.count_amino_acids()
        amino_acid_percent = prot_analysis.get_amino_acids_percent()
        
        return {
            'composition': amino_acid_composition,
            'percentage': amino_acid_percent,
            'total': len(protein_sequence.replace('*', ''))
        }
    except Exception as e:
        st.warning(f"Amino acid analysis error: {str(e)}")
        # Fallback to simple counting
        aa_count = Counter(protein_sequence.replace('*', ''))
        total = sum(aa_count.values())
        aa_percent = {aa: (count/total)*100 for aa, count in aa_count.items()}
        
        return {
            'composition': aa_count,
            'percentage': aa_percent,
            'total': total
        }

def load_enzymes_from_json():
    """Load restriction enzymes from JSON files in enzymes folder."""
    enzymes_data = {}
    enzymes_folder = "enzymes"
    
    if not os.path.exists(enzymes_folder):
        st.warning(f"Enzymes folder '{enzymes_folder}' not found. Creating example data.")
        # Create example enzymes data
        enzymes_data = {
            "Type II": [
                {"name": "EcoRI", "recognition_site": "GAATTC", "cut_position": 1},
                {"name": "BamHI", "recognition_site": "GGATCC", "cut_position": 1},
                {"name": "HindIII", "recognition_site": "AAGCTT", "cut_position": 1}
            ],
            "Type IIS": [
                {"name": "FokI", "recognition_site": "GGATG", "cut_position": 9},
                {"name": "BsaI", "recognition_site": "GGTCTC", "cut_position": 1}
            ]
        }
        return enzymes_data
    
    try:
        # Load all JSON files in the enzymes folder
        for file in os.listdir(enzymes_folder):
            if file.endswith('.json'):
                enzyme_type = file.replace('.json', '')
                file_path = os.path.join(enzymes_folder, file)
                
                with open(file_path, 'r') as f:
                    enzymes_data[enzyme_type] = json.load(f)
        
        return enzymes_data
    except Exception as e:
        st.error(f"Error loading enzymes: {str(e)}")
        return {}

def find_restriction_sites(sequence, enzymes):
    """Find restriction sites in the DNA sequence."""
    sites = []
    
    for enzyme in enzymes:
        rec_site = enzyme.get('recognition_site', '').upper()
        if not rec_site:
            continue
        
        # Find all occurrences of the recognition site
        pattern = rec_site.replace('N', '.')
        matches = list(re.finditer(pattern, sequence, re.IGNORECASE))
        
        for match in matches:
            start_pos = match.start() + 1  # Convert to 1-based
            end_pos = match.end()
            
            sites.append({
                'enzyme': enzyme['name'],
                'type': enzyme.get('type', 'Unknown'),
                'recognition_site': rec_site,
                'position': start_pos,
                'cut_position': enzyme.get('cut_position', 0),
                'sequence': sequence[start_pos-1:end_pos]
            })
    
    # Sort by position
    sites.sort(key=lambda x: x['position'])
    return sites

def parse_fasta_file(file_content):
    """Parse FASTA file content and extract sequences."""
    sequences = []
    try:
        fasta_io = StringIO(file_content)
        
        for record in SeqIO.parse(fasta_io, "fasta"):
            sequences.append({
                'id': record.id,
                'description': record.description,
                'sequence': str(record.seq).upper(),
                'length': len(record.seq)
            })
        
        return sequences, None
    except Exception as e:
        return [], f"Error parsing FASTA file: {str(e)}"

def perform_alignment(seq1, seq2, match, mismatch, gap_open, gap_extend):
    """Performs Pairwise Alignment."""
    try:
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = match
        aligner.mismatch_score = mismatch
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend
        
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]
        return str(best_alignment), best_alignment.score, None
    except Exception as e:
        return "", 0, f"Alignment error: {str(e)}"

def generate_pdf_report_enhanced(sequence_data, orfs, amino_acid_data, restriction_sites, gc_content):
    """Generate enhanced PDF report with charts and tables."""
    try:
        buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.HexColor('#3498db')
        )
        
        # Title
        story.append(Paragraph("Generize SaaS - Sequence Analysis Report", title_style))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # ========== 1. Sequence Information ==========
        story.append(Paragraph("1. Sequence Information", heading_style))
        
        seq_info_data = [
            ['Parameter', 'Value'],
            ['Sequence Length', f"{sequence_data['length']} bp"],
            ['GC Content', f"{sequence_data['gc']:.2f}%"],
            ['Melting Temperature', f"{sequence_data['tm']:.2f} °C" if sequence_data['tm'] > 0 else "N/A"],
            ['Coding Ratio', f"{sequence_data['coding_ratio']:.2f}%"],
            ['Number of ORFs', f"{len(orfs)}"],
        ]
        
        seq_info_table = Table(seq_info_data, colWidths=[200, 200])
        seq_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(seq_info_table)
        story.append(Spacer(1, 12))
        
        # ========== 2. ORF Information ==========
        if orfs:
            story.append(Paragraph("2. ORF Information", heading_style))
            
            orf_data = [['Frame', 'Start', 'End', 'Length (bp)', 'Protein Length']]
            for orf in orfs[:10]:  # Show top 10 ORFs
                orf_data.append([
                    orf['frame'],
                    orf['start'],
                    orf['end'],
                    orf['length'],
                    orf['protein_length']
                ])
            
            orf_table = Table(orf_data, colWidths=[60, 60, 60, 100, 100])
            orf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(orf_table)
            
            if len(orfs) > 10:
                story.append(Paragraph(f"... and {len(orfs) - 10} more ORFs", styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # ========== 3. Amino Acid Analysis ==========
        if amino_acid_data:
            story.append(Paragraph("3. Amino Acid Composition", heading_style))
            
            aa_data = [['Amino Acid', 'Count', 'Percentage']]
            for aa, count in amino_acid_data['composition'].items():
                percentage = amino_acid_data['percentage'].get(aa, 0)
                aa_data.append([aa, count, f"{percentage:.2f}%"])
            
            aa_table = Table(aa_data, colWidths=[80, 80, 100])
            aa_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(aa_table)
            story.append(Spacer(1, 12))
        
        # ========== 4. Restriction Sites ==========
        if restriction_sites:
            story.append(Paragraph("4. Restriction Sites Found", heading_style))
            
            restriction_data = [['Enzyme', 'Site', 'Position', 'Cut Position']]
            for site in restriction_sites[:15]:  # Show top 15 sites
                restriction_data.append([
                    site['enzyme'],
                    site['recognition_site'],
                    site['position'],
                    site['cut_position']
                ])
            
            restriction_table = Table(restriction_data, colWidths=[80, 80, 80, 80])
            restriction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(restriction_table)
            
            if len(restriction_sites) > 15:
                story.append(Paragraph(f"... and {len(restriction_sites) - 15} more sites", styles['Normal']))
            
            story.append(Spacer(1, 12))
        
        # ========== 5. Sequence Preview ==========
        story.append(Paragraph("5. Sequence Preview", heading_style))
        
        seq_preview = sequence_data['sequence'][:500]
        if len(sequence_data['sequence']) > 500:
            seq_preview += "..."
        
        story.append(Paragraph(f"First 500 bases:", styles['Normal']))
        story.append(Paragraph(f"<font face='Courier' size='9'>{seq_preview}</font>", 
                               ParagraphStyle('Code', parent=styles['Normal'])))
        
        # ========== 6. Footer ==========
        story.append(Spacer(1, 20))
        story.append(Paragraph("=" * 80, styles['Normal']))
        story.append(Paragraph(f"Generated by Generize SaaS v2.2.0 | {datetime.now().strftime('%Y-%m-%d')}", 
                               ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                            textColor=colors.grey)))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer, None
        
    except Exception as e:
        return None, f"PDF generation error: {str(e)}"

def fetch_pdb_structure(pdb_id):
    """Fetches PDB file content from RCSB with validation."""
    try:
        # Validate PDB ID format
        if not re.match(r'^[0-9a-zA-Z]{4}$', pdb_id.strip()):
            return None, "Invalid PDB ID format. Must be 4 characters (e.g., 1CRN)"
        
        pdb_id = pdb_id.strip().upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        return response.text, None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return None, f"PDB ID '{pdb_id}' not found."
        return None, f"HTTP Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ==========================================
# 4. UI COMPONENTS (Enhanced)
# ==========================================

def sidebar_nav():
    st.sidebar.title("🧬 Generize SaaS")
    st.sidebar.markdown("---")
    
    # Display version info
    st.sidebar.markdown("""
    **Version:** 2.2.0 (Enhanced)
    **License:** ✅ Active
    **Valid until:** 2026-10-30
    """)
    
    st.sidebar.markdown("---")
    
    return st.sidebar.radio("Navigation", 
        ["Dashboard", "Sequence Analysis", "Pairwise Alignment", "PDB Structure Viewer", "NCBI BLAST", "About"]
    )

def render_dashboard():
    st.title("Welcome to Generize SaaS")
    st.markdown("### Advanced Bioinformatics Platform")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🧬 **Sequence Analysis**\n\nEnhanced with ORF detection, restriction sites, and FASTA support.")
    with col2:
        st.success("🔗 **Alignment Tools**\n\nPerform Global/Local pairwise alignments with custom scoring.")
    with col3:
        st.warning("🧊 **3D Visualization**\n\nView molecular structures directly from PDB IDs or predict structures.")
    with col4:
        st.error("⚡ **New Features**\n\n- ORF Detection\n- Restriction Enzymes\n- Enhanced PDF Reports")
    
    st.markdown("---")
    
    # Try to load image with fallback
    try:
        st.image(
            "https://images.unsplash.com/photo-1530026405186-ed1f139313f8?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
            caption="Genomic Research", 
            use_column_width=True
        )
    except:
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; text-align: center;'>
        <h4>🔬 Bioinformatics Platform</h4>
        <p>Advanced tools for genomic analysis and molecular visualization</p>
        </div>
        """, unsafe_allow_html=True)

def render_sequence_analysis():
    st.header("🧬 DNA Sequence Analysis")
    
    # File upload or manual input
    input_method = st.radio("Input Method", ["Manual Input", "FASTA File Upload"], horizontal=True)
    
    sequence = ""
    fasta_info = None
    
    if input_method == "Manual Input":
        seq_input = st.text_area(
            "Enter DNA Sequence (FASTA or Raw)", 
            height=150, 
            help="Supported characters: A, T, C, G, N (case insensitive)",
            placeholder=">Sequence1\nATCGATCGATCG...\n\nOr raw sequence: ATCGATCG..."
        )
        if seq_input:
            # Check if it's FASTA format
            if seq_input.startswith('>'):
                sequences, error = parse_fasta_file(seq_input)
                if error:
                    st.error(error)
                elif sequences:
                    fasta_info = sequences[0]
                    sequence = fasta_info['sequence']
                    st.success(f"Loaded sequence: {fasta_info['id']} ({fasta_info['length']} bp)")
            else:
                sequence = clean_sequence(seq_input)
    
    else:  # FASTA File Upload
        uploaded_file = st.file_uploader("Choose a FASTA file", type=['fasta', 'fa', 'txt'])
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            sequences, error = parse_fasta_file(file_content)
            
            if error:
                st.error(error)
            elif sequences:
                # Let user select sequence if multiple
                if len(sequences) > 1:
                    seq_options = [f"{seq['id']} ({seq['length']} bp)" for seq in sequences]
                    selected = st.selectbox("Select Sequence", seq_options)
                    selected_index = seq_options.index(selected)
                    fasta_info = sequences[selected_index]
                else:
                    fasta_info = sequences[0]
                
                sequence = fasta_info['sequence']
                st.success(f"Loaded: {fasta_info['id']} - {fasta_info['description']}")
                st.info(f"Length: {fasta_info['length']} bp")
    
    if sequence:
        clean_seq = clean_sequence(sequence)
        
        # Validate sequence
        is_valid, validation_msg = validate_dna_sequence(clean_seq)
        
        if not is_valid:
            st.markdown(f'<div class="error-box">{validation_msg}</div>', unsafe_allow_html=True)
        
        # Basic Info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Sequence Length", f"{len(clean_seq)} bp")
        with col_info2:
            if fasta_info:
                st.metric("Sequence ID", fasta_info['id'])
        
        # Restriction Enzymes Section
        st.subheader("🔪 Restriction Enzyme Analysis")
        
        # Load enzymes
        enzymes_data = load_enzymes_from_json()
        
        if enzymes_data:
            enzyme_categories = list(enzymes_data.keys())
            
            col_enzyme1, col_enzyme2 = st.columns(2)
            with col_enzyme1:
                selected_category = st.selectbox("Select Enzyme Category", enzyme_categories)
            
            if selected_category and selected_category in enzymes_data:
                enzymes = enzymes_data[selected_category]
                
                # Display selected enzymes
                with col_enzyme2:
                    min_site_length = st.slider("Minimum site length", 4, 10, 6)
                
                # Filter enzymes by minimum recognition site length
                filtered_enzymes = [e for e in enzymes if len(e.get('recognition_site', '')) >= min_site_length]
                
                if st.button("Scan for Restriction Sites", type="secondary"):
                    with st.spinner("Scanning for restriction sites..."):
                        restriction_sites = find_restriction_sites(clean_seq, filtered_enzymes)
                        
                        if restriction_sites:
                            st.markdown(f'<div class="success-box">Found {len(restriction_sites)} restriction sites</div>', 
                                       unsafe_allow_html=True)
                            
                            # Display restriction sites
                            sites_df = pd.DataFrame(restriction_sites)
                            st.dataframe(
                                sites_df[['enzyme', 'recognition_site', 'position', 'cut_position']],
                                use_container_width=True,
                                height=200
                            )
                            
                            # Save for PDF report
                            st.session_state.restriction_sites = restriction_sites
                        else:
                            st.info("No restriction sites found for selected enzymes.")
                            st.session_state.restriction_sites = []
        
        # Main Analysis Button
        if st.button("🔬 Run Complete Analysis", type="primary"):
            with st.spinner("Performing comprehensive analysis..."):
                # Calculate basic metrics
                gc, tm, protein, error = calculate_dna_metrics(clean_seq)
                
                if error:
                    st.error(error)
                else:
                    # Find ORFs
                    orfs = find_orfs(clean_seq, min_orf_length=30)
                    
                    # Calculate coding ratio
                    coding_ratio = calculate_coding_ratio(clean_seq, orfs)
                    
                    # Analyze amino acids
                    amino_acid_data = analyze_amino_acids(protein)
                    
                    # Store data in session state for PDF
                    sequence_data = {
                        'sequence': clean_seq,
                        'length': len(clean_seq),
                        'gc': gc,
                        'tm': tm,
                        'protein': protein,
                        'coding_ratio': coding_ratio
                    }
                    
                    st.session_state.sequence_data = sequence_data
                    st.session_state.orfs = orfs
                    st.session_state.amino_acid_data = amino_acid_data
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Metrics Display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("GC Content", f"{gc:.2f}%")
                    with col2:
                        if tm > 0:
                            st.metric("Melting Temp (Tm)", f"{tm:.2f} °C")
                        else:
                            st.metric("Melting Temp (Tm)", "N/A")
                    with col3:
                        st.metric("Coding Ratio", f"{coding_ratio:.2f}%")
                    with col4:
                        st.metric("ORFs Found", len(orfs))
                    
                    # ORF Display
                    st.subheader("🧬 ORF Detection")
                    if orfs:
                        orf_df = pd.DataFrame(orfs)
                        st.dataframe(
                            orf_df[['frame', 'start', 'end', 'length', 'protein_length']],
                            use_container_width=True,
                            height=300
                        )
                        
                        # Show longest ORF protein
                        if orfs:
                            longest_orf = orfs[0]
                            with st.expander(f"View Longest ORF Protein (Frame {longest_orf['frame']}, {longest_orf['protein_length']} aa)"):
                                st.code(longest_orf['protein'], language="text")
                    else:
                        st.info("No ORFs found (minimum length: 30 bp)")
                    
                    # Amino Acid Analysis
                    st.subheader("🧪 Amino Acid Composition")
                    if amino_acid_data:
                        col_aa1, col_aa2 = st.columns(2)
                        
                        with col_aa1:
                            aa_df = pd.DataFrame({
                                'Amino Acid': list(amino_acid_data['composition'].keys()),
                                'Count': list(amino_acid_data['composition'].values()),
                                'Percentage': [f"{p:.2f}%" for p in amino_acid_data['percentage'].values()]
                            })
                            st.dataframe(aa_df, use_container_width=True, height=300)
                        
                        with col_aa2:
                            # Create amino acid distribution chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(amino_acid_data['composition'].keys()),
                                    y=list(amino_acid_data['composition'].values()),
                                    marker_color='lightblue'
                                )
                            ])
                            fig.update_layout(
                                title="Amino Acid Distribution",
                                xaxis_title="Amino Acid",
                                yaxis_title="Count",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Protein Translation
                    st.subheader("💊 Protein Translation")
                    st.code(protein[:500] + ("..." if len(protein) > 500 else ""), language="text")
                    
                    # Generate Enhanced PDF Report
                    st.subheader("📥 Download Report")
                    
                    restriction_sites = st.session_state.get('restriction_sites', [])
                    
                    pdf_bytes, pdf_error = generate_pdf_report_enhanced(
                        sequence_data, orfs, amino_acid_data, restriction_sites, gc
                    )
                    
                    if pdf_error:
                        st.error(f"PDF generation failed: {pdf_error}")
                        
                        # Fallback to text report
                        report_text = f"""Sequence Analysis Report
========================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sequence Length: {len(clean_seq)} bp
GC Content: {gc:.2f}%
Coding Ratio: {coding_ratio:.2f}%
Melting Temperature: {tm:.2f} °C
ORFs Found: {len(orfs)}

TOP 5 ORFs:
"""
                        for i, orf in enumerate(orfs[:5], 1):
                            report_text += f"{i}. Frame {orf['frame']}: {orf['start']}-{orf['end']} ({orf['length']} bp)\n"
                        
                        report_text += f"\nPROTEIN TRANSLATION:\n{protein[:1000]}"
                        
                        st.download_button(
                            label="Download Text Report",
                            data=report_text,
                            file_name="sequence_analysis.txt",
                            mime="text/plain"
                        )
                    else:
                        st.download_button(
                            label="📥 Download Enhanced PDF Report",
                            data=pdf_bytes,
                            file_name="sequence_analysis_report.pdf",
                            mime="application/pdf",
                            type="primary"
                        )

def render_alignment():
    # ... (keep existing alignment code) ...
    st.header("🔗 Pairwise Sequence Alignment")
    
    col1, col2 = st.columns(2)
    with col1:
        seq1 = st.text_area("Sequence A", height=100, placeholder="Enter first DNA sequence...")
    with col2:
        seq2 = st.text_area("Sequence B", height=100, placeholder="Enter second DNA sequence...")
    
    with st.expander("⚙️ Alignment Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        match = c1.number_input("Match Score", value=2.0, step=0.5)
        mismatch = c2.number_input("Mismatch Score", value=-1.0, step=0.5)
        gap_open = c3.number_input("Gap Open Penalty", value=-2.0, step=0.5)
        gap_extend = c4.number_input("Gap Extend Penalty", value=-0.5, step=0.1)
    
    if st.button("Run Alignment", type="primary"):
        if not seq1 or not seq2:
            st.warning("Please enter both sequences.")
        else:
            s1 = clean_sequence(seq1)
            s2 = clean_sequence(seq2)
            
            # Validate sequences
            valid1, msg1 = validate_dna_sequence(s1)
            valid2, msg2 = validate_dna_sequence(s2)
            
            if not valid1:
                st.error(f"Sequence A: {msg1}")
            if not valid2:
                st.error(f"Sequence B: {msg2}")
            
            if valid1 and valid2:
                with st.spinner("Performing alignment..."):
                    alignment_str, score, error = perform_alignment(
                        s1, s2, match, mismatch, gap_open, gap_extend
                    )
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Alignment completed! Score: {score:.2f}")
                        
                        # Display stats
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Sequence A Length", f"{len(s1)} bp")
                        with col_stat2:
                            st.metric("Sequence B Length", f"{len(s2)} bp")
                        with col_stat3:
                            st.metric("Alignment Score", f"{score:.2f}")
                        
                        st.subheader("Alignment Result")
                        
                        # Display alignment in a nice format
                        alignment_lines = alignment_str.split('\n')
                        if len(alignment_lines) >= 3:
                            with st.container():
                                st.text("Sequence A: " + alignment_lines[0][:100] + ("..." if len(alignment_lines[0]) > 100 else ""))
                                st.text("           " + alignment_lines[1][:100] + ("..." if len(alignment_lines[1]) > 100 else ""))
                                st.text("Sequence B: " + alignment_lines[2][:100] + ("..." if len(alignment_lines[2]) > 100 else ""))
                            
                            with st.expander("View Full Alignment"):
                                st.text(alignment_str)
                        else:
                            st.text(alignment_str)

def render_pdb_viewer():
    st.header("🧊 3D Molecular Viewer & Structure Prediction")
    
    tab1, tab2 = st.tabs(["View Existing PDB", "Predict Structure"])
    
    with tab1:
        col_input, col_style = st.columns([2, 1])
        with col_input:
            pdb_id = st.text_input("Enter PDB ID", value="1CRN", 
                                 help="Enter a valid 4-character PDB ID (e.g., 1CRN, 6VXX, 2HHB)")
        with col_style:
            style = st.selectbox("Visualization Style", 
                               ["cartoon", "stick", "sphere", "line", "cross", "surface"])
        
        # Example PDB IDs
        with st.expander("💡 Example PDB IDs"):
            examples = st.columns(5)
            example_ids = ["1CRN", "6VXX", "2HHB", "1UBQ", "1TLD"]
            for i, (col, pid) in enumerate(zip(examples, example_ids)):
                with col:
                    if st.button(pid, key=f"pdb_ex_{i}"):
                        st.session_state.pdb_id_example = pid
        
        # Check if example was clicked
        if 'pdb_id_example' in st.session_state:
            pdb_id = st.session_state.pdb_id_example
            del st.session_state.pdb_id_example
        
        if st.button("Load Structure", type="primary"):
            if not pdb_id:
                st.warning("Please enter a PDB ID.")
            else:
                with st.spinner(f"Fetching structure {pdb_id.upper()}..."):
                    pdb_data, error = fetch_pdb_structure(pdb_id)
                    
                    if error:
                        st.error(error)
                    else:
                        st.subheader(f"Structure: {pdb_id.upper()}")
                        
                        # Display structure info
                        st.info(f"Structure loaded successfully ({len(pdb_data.splitlines())} lines)")
                        
                        # Try to render 3D view
                        try:
                            if 'py3Dmol' in globals() and 'showmol' in globals():
                                xyzview = py3Dmol.view(width=800, height=500)
                                xyzview.addModel(pdb_data, 'pdb')
                                
                                # Set style based on selection
                                if style == "cartoon":
                                    xyzview.setStyle({'cartoon': {'color': 'spectrum'}})
                                elif style == "surface":
                                    xyzview.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})
                                    xyzview.setStyle({'stick': {}})
                                else:
                                    xyzview.setStyle({style: {'color': 'spectrum'}})
                                
                                xyzview.zoomTo()
                                showmol(xyzview, height=500, width=800)
                            else:
                                st.warning("3D visualization libraries not available. Displaying raw PDB data.")
                                with st.expander("View PDB File Content"):
                                    st.text(pdb_data[:2000] + ("..." if len(pdb_data) > 2000 else ""))
                        except Exception as e:
                            st.error(f"3D visualization error: {str(e)}")
                            with st.expander("View PDB File Content"):
                                st.text(pdb_data[:1000])
    
    with tab2:
        st.subheader("🔮 Structure Prediction")
        st.markdown("""
        <div class="warning-box">
        <strong>Note:</strong> This is a simulation module. In production, this would connect to 
        structure prediction services like AlphaFold, RoseTTAFold, or I-TASSER.
        </div>
        """, unsafe_allow_html=True)
        
        pred_method = st.selectbox("Prediction Method", 
                                  ["AlphaFold (Simulated)", "I-TASSER (Simulated)", "Local Prediction"])
        
        # Sequence input for prediction
        pred_input = st.text_area("Enter Protein Sequence for Structure Prediction", 
                                 height=100,
                                 placeholder="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ...")
        
        if st.button("Predict Structure", type="primary"):
            if not pred_input:
                st.warning("Please enter a protein sequence.")
            else:
                with st.spinner(f"Running {pred_method} prediction..."):
                    # Simulate prediction process
                    import time
                    time.sleep(2)
                    
                    st.success("Prediction complete!")
                    
                    # Display simulated results
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        st.metric("Sequence Length", f"{len(pred_input)} aa")
                        st.metric("Predicted Confidence", "85%")
                        st.metric("Estimated Time (real)", "2-4 hours")
                    
                    with col_pred2:
                        st.info("""
                        **Simulated Prediction Results:**
                        
                        - **pLDDT Score**: 85.4
                        - **Secondary Structure**: 
                          * Alpha-helix: 45%
                          * Beta-sheet: 25%
                          * Coil: 30%
                        - **Predicted Domains**: 2
                        """)
                    
                    # Simulated 3D visualization
                    st.subheader("Predicted 3D Structure (Simulated)")
                    st.info("In production, this would display the actual predicted structure.")

def render_ncbi_blast():
    st.header("🌍 NCBI Tool Connector")
    
    st.markdown("""
    <div class="warning-box">
    <strong>Note:</strong> This is a simulation module. In production, this would connect to 
    actual NCBI APIs using Biopython's Entrez and NCBIWWQ modules.
    </div>
    """, unsafe_allow_html=True)
    
    tool = st.selectbox("Select Tool", 
                       ["BLAST (Nucleotide)", "BLAST (Protein)", "PubMed Search", "Fetch GenBank Record"])
    
    query = st.text_input("Enter Query / Accession ID", 
                         placeholder="e.g., NM_001301717, cancer therapy, P04637")
    
    # Additional parameters based on tool
    if "BLAST" in tool:
        col_db, col_evalue = st.columns(2)
        with col_db:
            database = st.selectbox("Database", ["nr", "refseq_rna", "refseq_protein", "pdb"])
        with col_evalue:
            evalue = st.number_input("E-value threshold", value=0.001, format="%.3f")
    
    if st.button("Execute Search", type="primary"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Connecting to NCBI..."):
                # Simulate API call delay
                import time
                time.sleep(1.5)
                
                st.success("Connection established!")
                
                if "BLAST" in tool:
                    st.markdown(f"""
                    **BLAST Simulation Results for:** `{query}`
                    
                    - Database: `{database}`
                    - E-value threshold: `{evalue}`
                    - Tool: `{tool}`
                    
                    ### Simulated Results:
                    1. **Top Hit**: Hypothetical protein (Accession: XP_123456.1)
                       - E-value: 2e-45
                       - Identity: 98%
                       - Coverage: 100%
                    
                    2. **Second Hit**: Conserved domain protein (Accession: NP_987654.1)
                       - E-value: 1e-30
                       - Identity: 85%
                       - Coverage: 95%
                    
                    *Note: In production, this would show actual BLAST results using Biopython's qblast()*
                    """)
                else:
                    st.markdown(f"""
                    **Search Results for:** `{query}`
                    
                    ### Simulated Results:
                    1. **Article Title**: Advances in genomic sequencing
                       - Authors: Smith J, et al.
                       - Journal: Nature Genetics, 2023
                       - PMID: 12345678
                    
                    2. **Article Title**: CRISPR-based therapeutics
                       - Authors: Johnson A, et al.
                       - Journal: Science, 2022
                       - PMID: 87654321
                    
                    *Total results: 245 papers found*
                    """)

def render_about():
    st.header("About Generize SaaS v2.2.0")
    
    st.markdown("""
    **Generize SaaS** is a modernized bioinformatics platform originally built as a desktop utility. 
    It leverages the power of Biopython for accurate scientific computation and provides an intuitive 
    web interface for researchers and students.
    
    ### 🎯 New Features in v2.2.0
    - **ORF Detection**: Find open reading frames from start to stop codons
    - **Coding Ratio Analysis**: Calculate percentage of coding nucleotides
    - **Enhanced PDF Reports**: Professional reports with tables and formatting
    - **Restriction Enzyme Analysis**: Scan for enzyme cutting sites
    - **FASTA File Support**: Upload and parse FASTA format files
    - **Structure Prediction**: Simulated protein structure prediction
    
    ### 🔧 Technology Stack
    - **Backend**: Python, Biopython, Streamlit
    - **Visualization**: Py3Dmol, Plotly, Matplotlib
    - **Reporting**: ReportLab (Enhanced PDF generation)
    - **Data Processing**: Pandas, NumPy
    
    ### 📊 License Status
    - ✅ **License Active**
    - 📅 **Valid until**: 2026-10-30
    - 👥 **User Limit**: Unlimited (SaaS Edition)
    
    ### 📞 Support
    For technical support or feature requests, please contact:
    - Email: support@generize.com
    - Documentation: docs.generize.com
    - GitHub: github.com/generize/saas
    
    **Version:** 2.2.0 (Enhanced Edition)
    **Last Updated:** December 2023
    """)
    
    # Display system status
    st.markdown("---")
    st.subheader("System Status")
    
    status_cols = st.columns(5)
    
    with status_cols[0]:
        st.metric("Biopython", "✅" if 'Bio' in globals() else "❌")
    with status_cols[1]:
        st.metric("ORF Detection", "✅")
    with status_cols[2]:
        st.metric("3D Visualization", "✅" if 'py3Dmol' in globals() else "❌")
    with status_cols[3]:
        st.metric("Enhanced PDF", "✅")
    with status_cols[4]:
        st.metric("FASTA Support", "✅")

# ==========================================
# 5. MAIN APP ROUTER
# ==========================================

def main():
    # Check if libraries are loaded
    if not LIBRARIES_OK:
        st.error("Critical libraries are missing. Please check the installation.")
        st.stop()
    
    try:
        # Initialize session state
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        if 'restriction_sites' not in st.session_state:
            st.session_state.restriction_sites = []
        
        page = sidebar_nav()
        
        # Page routing
        if page == "Dashboard":
            render_dashboard()
        elif page == "Sequence Analysis":
            render_sequence_analysis()
        elif page == "Pairwise Alignment":
            render_alignment()
        elif page == "PDB Structure Viewer":
            render_pdb_viewer()
        elif page == "NCBI BLAST":
            render_ncbi_blast()
        elif page == "About":
            render_about()
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.markdown("""
        <div class="error-box">
        <strong>Application Error</strong><br>
        The application encountered an unexpected error. Please try refreshing the page.
        If the problem persists, contact support.
        </div>
        """, unsafe_allow_html=True)
        
        # Option to show error details
        if st.checkbox("Show error details (for debugging)"):
            st.exception(e)

if __name__ == "__main__":
    main()