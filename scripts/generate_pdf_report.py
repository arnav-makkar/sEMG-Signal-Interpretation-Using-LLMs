#!/usr/bin/env python3
"""
Clinical PDF Report Generator
==============================
Builds a physician-facing PDF report for each participant and an optional
combined report. Content per participant:
  - Cover page with clinical assessment
  - Session overview
  - Per-sensor activation table
  - Cross-channel analysis
  - LLM clinical interpretation
  - Visualizations (activation map, RMS trends, co-contraction)
  - Methodology note

Output: results/reports/P{N}_activation_report.pdf
        results/reports/All_Participants_Report.pdf  (combined)
"""

import json
import os
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


PARTICIPANTS = [f'P{i}' for i in range(1, 11)]
DATA_DIR = 'DATA'
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'activation_plots')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

# ==============================
# STYLES
# ==============================

def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ClinicalTitle',
        parent=styles['Title'],
        fontSize=20, leading=24, textColor=colors.HexColor('#1a3a6c'),
        spaceAfter=12, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontSize=12, leading=16, textColor=colors.HexColor('#555555'),
        spaceAfter=18, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=14, leading=18, textColor=colors.HexColor('#1a3a6c'),
        spaceBefore=14, spaceAfter=8,
        borderWidth=0, borderPadding=0,
        leftIndent=0, borderColor=colors.HexColor('#1a3a6c')
    ))
    styles.add(ParagraphStyle(
        name='SubSection',
        parent=styles['Heading2'],
        fontSize=11, leading=14, textColor=colors.HexColor('#2c5282'),
        spaceBefore=10, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='BodyJustify',
        parent=styles['BodyText'],
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='KeyFinding',
        parent=styles['BodyText'],
        fontSize=10, leading=14,
        backColor=colors.HexColor('#fffbea'),
        borderColor=colors.HexColor('#e5c07b'),
        borderWidth=1, borderPadding=8,
        spaceAfter=8, spaceBefore=4
    ))
    styles.add(ParagraphStyle(
        name='ClinicalNote',
        parent=styles['BodyText'],
        fontSize=10, leading=14,
        backColor=colors.HexColor('#f0f7ff'),
        borderColor=colors.HexColor('#4a90d9'),
        borderWidth=1, borderPadding=8,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='SmallNote',
        parent=styles['BodyText'],
        fontSize=8, leading=10, textColor=colors.HexColor('#666666'),
        alignment=TA_JUSTIFY, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='CoverKey',
        parent=styles['BodyText'],
        fontSize=11, leading=15, alignment=TA_CENTER,
        spaceAfter=4
    ))
    return styles


STYLES = make_styles()


# ==============================
# HELPERS
# ==============================

def load_data(participant):
    """Load all data for a participant."""
    base = os.path.join(DATA_DIR, participant)
    data = {
        'participant': participant,
        'features': None,
        'result': None,
        'plot_path': os.path.join(PLOTS_DIR, f'{participant}_activation.png'),
    }

    feat_path = os.path.join(base, 'activation_features.json')
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            data['features'] = json.load(f)

    res_path = os.path.join(base, 'activation_result.json')
    if os.path.exists(res_path):
        with open(res_path) as f:
            data['result'] = json.load(f)

    return data


def fmt_num(x, digits=3):
    if x is None:
        return 'N/A'
    try:
        return f"{float(x):.{digits}f}"
    except (ValueError, TypeError):
        return str(x)


def grade_color(level):
    """Color for activation level."""
    mapping = {
        'low':       colors.HexColor('#3b82f6'),
        'moderate':  colors.HexColor('#10b981'),
        'high':      colors.HexColor('#f59e0b'),
        'veryhigh':  colors.HexColor('#ef4444'),
        'very_high': colors.HexColor('#ef4444'),
    }
    return mapping.get(str(level).lower(), colors.HexColor('#6b7280'))


# ==============================
# REPORT SECTIONS
# ==============================

def build_cover(participant, features, result):
    """Cover page with executive clinical summary."""
    story = []
    story.append(Spacer(1, 1.5 * cm))

    story.append(Paragraph("sEMG Muscle Activation Report", STYLES['ClinicalTitle']))
    story.append(Paragraph(
        f"Participant {participant} &nbsp;·&nbsp; Gesture_1 &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%B %d, %Y')}",
        STYLES['Subtitle']))

    story.append(Spacer(1, 0.8 * cm))

    # Key findings box
    if result and 'parse_error' not in result:
        overall = str(result.get('overall_activation', 'N/A')).replace('_', ' ').title()
        profile = str(result.get('temporal_profile', 'N/A'))
        consistency = str(result.get('activation_consistency', 'N/A')).title()
        dominant = ', '.join(result.get('dominant_sensors', []))
        weakest = ', '.join(result.get('weakest_sensors', []))
        confidence = result.get('confidence', 'N/A')

        cover_data = [
            ['Overall Activation', overall],
            ['Temporal Profile', profile],
            ['Consistency', consistency],
            ['Dominant Sensors', dominant],
            ['Weakest Sensors', weakest],
            ['Assessment Confidence', f"{confidence}" if confidence != 'N/A' else 'N/A'],
        ]
        t = Table(cover_data, colWidths=[5.5 * cm, 8 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1a3a6c')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#cccccc')),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f5f7fa')),
        ]))
        story.append(t)

        story.append(Spacer(1, 0.6 * cm))

        notes = result.get('clinical_notes', '')
        if notes:
            story.append(Paragraph(
                f"<b>Clinical Summary:</b> {notes}",
                STYLES['ClinicalNote']))

        flags = result.get('flags', [])
        if flags:
            flags_html = '<br/>'.join([f"• {f}" for f in flags])
            story.append(Paragraph(
                f"<b>Key Findings:</b><br/>{flags_html}",
                STYLES['KeyFinding']))

    story.append(Spacer(1, 0.8 * cm))

    # Session stats
    if features:
        duration = features.get('duration_s', 'N/A')
        n_sensors = features.get('n_sensors', 'N/A')
        fs = features.get('sampling_rate_hz', 'N/A')
        rms = features.get('session_mean_rms', 'N/A')

        stats_data = [
            ['Session Duration', f"{duration} s ({duration/60:.1f} min)" if isinstance(duration, (int, float)) else str(duration)],
            ['Sensors', f"{n_sensors} (S1–S4 flexor, S5–S8 extensor)"],
            ['Sampling Rate', f"{fs} Hz" if fs != 'N/A' else 'N/A'],
            ['Session Mean RMS', f"{fmt_num(rms, 6)} mV"],
            ['Feature Source', features.get('feature_source', 'N/A')],
        ]
        t = Table(stats_data, colWidths=[5.5 * cm, 8 * cm])
        t.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("Session Overview", STYLES['SubSection']))
        story.append(t)

    story.append(PageBreak())
    return story


def build_per_sensor_section(features, result):
    """Per-sensor activation table."""
    story = []
    story.append(Paragraph("Per-Sensor Activation Analysis", STYLES['SectionHeader']))

    story.append(Paragraph(
        "Muscle activity was recorded across 8 sensors. Sensors S1–S4 are positioned "
        "on the palmar/flexor compartment and S5–S8 on the dorsal/extensor compartment. "
        "RMS (Root Mean Square) reflects overall activation intensity; MNF and MDF "
        "(Mean and Median Frequency) characterise the spectral content of the EMG signal; "
        "WAMP (Willison Amplitude) indicates motor-unit firing activity.",
        STYLES['BodyJustify']))

    sensors = features['sensors']
    llm_per_sensor = {s['sensor']: s for s in result.get('per_sensor', [])} \
        if result and 'per_sensor' in result else {}

    header = ['Sensor', 'Mean RMS (mV)', 'CV', 'MNF (Hz)', 'MDF (Hz)', 'WAMP', 'Level', 'Profile']
    rows = [header]

    for sid in [f'S{i}' for i in range(1, 9)]:
        s = sensors[sid]
        llm = llm_per_sensor.get(sid, {})
        level = str(llm.get('activation_level', s['activation_level']))
        profile = str(llm.get('temporal_profile', s['temporal_profile']))
        rows.append([
            sid,
            fmt_num(s['mean_rms'], 6),
            fmt_num(s['cv_rms'], 3),
            fmt_num(s['mean_mnf'], 1),
            fmt_num(s['mean_mdf'], 1),
            fmt_num(s['mean_wamp'], 0),
            level.replace('_', ' ').title(),
            profile,
        ])

    t = Table(rows, colWidths=[1.3*cm, 2.4*cm, 1.5*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2.3*cm, 2.3*cm])
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a6c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f9fafb')]),
    ]

    # Color-code flexor vs extensor rows
    for i in range(1, 5):  # S1-S4 flexor
        style.append(('BACKGROUND', (0, i), (0, i), colors.HexColor('#dbeafe')))
    for i in range(5, 9):  # S5-S8 extensor
        style.append(('BACKGROUND', (0, i), (0, i), colors.HexColor('#fce7f3')))

    t.setStyle(TableStyle(style))
    story.append(t)

    story.append(Paragraph(
        "<i>Flexor sensors (S1–S4) shown with blue labels; extensor sensors (S5–S8) with pink.</i>",
        STYLES['SmallNote']))

    story.append(Spacer(1, 0.3 * cm))

    # Per-sensor clinical notes from LLM
    notable = [(sid, llm_per_sensor[sid].get('notable'))
               for sid in llm_per_sensor if llm_per_sensor[sid].get('notable')]
    if notable:
        story.append(Paragraph("Notable Findings per Sensor", STYLES['SubSection']))
        for sid, note in notable:
            story.append(Paragraph(f"<b>{sid}:</b> {note}", STYLES['BodyJustify']))

    return story


def build_cross_channel_section(features, result):
    """Cross-channel / spatial analysis."""
    story = []
    story.append(Paragraph("Spatial Activation Patterns", STYLES['SectionHeader']))

    cc = features['cross_channel']
    llm_cc = result.get('cross_channel', {}) if result else {}

    story.append(Paragraph(
        "Cross-channel metrics characterise how muscle activation is distributed across the "
        "forearm during the recorded gesture. A <b>high concentration index</b> indicates focal "
        "activation (one or two dominant sensors); a <b>low value</b> suggests diffuse activation. "
        "The <b>flexor/extensor balance</b> reveals which compartment drives the gesture. "
        "<b>Pattern stability</b> near 1.0 indicates the spatial pattern is maintained throughout "
        "the session, with no substantial load redistribution.",
        STYLES['BodyJustify']))

    cc_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Dominant Sensor', f"S{cc['dominant_sensor']}", 'Primary contributor'],
        ['Weakest Sensor', f"S{cc['weakest_sensor']}", 'Lowest activity'],
        ['Dominant-to-Weakest Ratio', f"{cc['dominant_weakest_ratio']:.2f}×",
         'Activation disparity'],
        ['Concentration Index', fmt_num(cc['mean_concentration_index']),
         'Focal (near 1) vs Diffuse (near 0.125)'],
        ['Pattern Stability', fmt_num(cc['mean_pattern_stability']),
         'Spatial consistency (1 = stable)'],
        ['Inter-Channel Correlation', fmt_num(cc['inter_channel_correlation']),
         'Crosstalk check (>0.8 = concern)'],
        ['Co-Contraction Ratio', fmt_num(cc.get('mean_co_contraction_ratio')),
         'Agonist/antagonist balance'],
        ['Flexor/Extensor Balance', fmt_num(cc.get('mean_flexor_extensor_balance'), 2),
         '>1 = flexor-dominant'],
        ['Overall Temporal Profile', cc['overall_temporal_profile'],
         'Session-wide activation trend'],
    ]
    t = Table(cc_data, colWidths=[5*cm, 2.5*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a6c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#333333')),
        ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#555555')),
        ('FONTSIZE', (2, 1), (2, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f9fafb')]),
    ]))
    story.append(t)

    # LLM-derived interpretations
    story.append(Spacer(1, 0.3 * cm))
    co_cont = llm_cc.get('co_contraction_likely')
    spat_comp = llm_cc.get('spatial_compensation')

    interp_lines = []
    if co_cont is not None:
        interp_lines.append(
            f"<b>Co-contraction:</b> {'Detected' if co_cont else 'Not detected'} — "
            f"{'Agonist and antagonist muscles activating simultaneously (normal for power grip).' if co_cont else 'Reciprocal activation pattern typical of efficient single-direction movement.'}"
        )
    if spat_comp is not None:
        interp_lines.append(
            f"<b>Spatial Compensation:</b> {'Detected' if spat_comp else 'Not detected'} — "
            f"{'Load redistribution across sensors observed, which may indicate motor adaptation.' if spat_comp else 'Stable spatial activation pattern throughout the session.'}"
        )

    for line in interp_lines:
        story.append(Paragraph(line, STYLES['BodyJustify']))

    gesture_notes = result.get('gesture_pattern_notes', '') if result else ''
    if gesture_notes:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"<b>Gesture-Pattern Interpretation:</b> {gesture_notes}",
            STYLES['ClinicalNote']))

    return story


def build_mu_recruitment_section(result):
    """Motor unit recruitment findings."""
    story = []
    if not result:
        return story
    mur = result.get('motor_unit_recruitment_signs')
    if not mur:
        return story

    story.append(Paragraph("Motor-Unit Recruitment", STYLES['SectionHeader']))

    detected = mur.get('detected', False)
    sensors_affected = mur.get('sensors_affected', [])
    evidence = mur.get('evidence', '')

    story.append(Paragraph(
        "Motor-unit recruitment is inferred from the Willison Amplitude (WAMP) and zero-crossing "
        "rate. Elevated WAMP indicates frequent changes in signal amplitude, consistent with "
        "the firing of additional (often higher-threshold) motor units as effort or fatigue "
        "increases.",
        STYLES['BodyJustify']))

    status_data = [
        ['Recruitment Detected', 'Yes' if detected else 'No'],
        ['Sensors Affected', ', '.join(sensors_affected) if sensors_affected else '—'],
    ]
    t = Table(status_data, colWidths=[5*cm, 10.5*cm])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f5f7fa')),
    ]))
    story.append(t)

    if evidence:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"<b>Evidence:</b> {evidence}", STYLES['ClinicalNote']))

    return story


def build_visualization_section(plot_path):
    """Embed activation analysis plot."""
    story = []
    if not os.path.exists(plot_path):
        return story
    story.append(Paragraph("Visualisation", STYLES['SectionHeader']))
    story.append(Paragraph(
        "The figure below consolidates the session analysis: per-sensor RMS amplitudes, "
        "spatial activation map, RMS time series for dominant sensors, co-contraction ratio, "
        "and frequency-content (MNF/MDF) trends over the 20-minute session.",
        STYLES['BodyJustify']))

    try:
        img = Image(plot_path, width=16*cm, height=16*cm, kind='proportional')
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"<i>Plot could not be embedded: {e}</i>", STYLES['SmallNote']))
    return story


def build_methodology(features, result):
    """Methodology and limitations footer."""
    story = []
    story.append(PageBreak())
    story.append(Paragraph("Methodology & Limitations", STYLES['SectionHeader']))

    story.append(Paragraph("Signal Processing", STYLES['SubSection']))
    story.append(Paragraph(
        "Raw surface EMG signals from 8 Delsys Avanti sensors were band-pass filtered "
        "(20–450 Hz, Butterworth 4th order) to attenuate motion artefacts and high-frequency "
        "noise. Twelve features were extracted per sensor in consecutive 20-second windows: "
        "RMS, MAV, IEMG, Waveform Length, Zero Crossings, Slope Sign Changes, Willison Amplitude, "
        "Peak Amplitude, Variance, Mean Frequency, Median Frequency, and Total Power. "
        "Cross-channel metrics (activation map, concentration index, pattern stability, "
        "co-contraction ratio, flexor/extensor balance) were computed from the per-sensor "
        "RMS time series.",
        STYLES['BodyJustify']))

    story.append(Paragraph("Clinical Interpretation", STYLES['SubSection']))
    model_name = result.get('_meta', {}).get('model', 'LLM') if result else 'LLM'
    story.append(Paragraph(
        f"Feature summaries were passed to a large language model ({model_name}) along with "
        "a structured set of clinical rules (activation level thresholds, gesture patterns, "
        "co-contraction criteria, motor-unit recruitment indicators) and retrieval-augmented "
        "research context drawn from a knowledge base of sEMG literature "
        "(Konrad 2005, Phinyomark 2012, Vigotsky 2018, Nazmi 2016, Yang 2024 EMGBench).",
        STYLES['BodyJustify']))

    story.append(Paragraph("Limitations", STYLES['SubSection']))
    story.append(Paragraph(
        "• <b>No %MVC normalisation:</b> Amplitudes are absolute (mV); direct cross-subject "
        "comparisons of amplitude are unreliable without a reference contraction.<br/>"
        "• <b>Sustained-contraction data:</b> Because the gesture is held continuously, onset "
        "and duty-cycle metrics are not reported.<br/>"
        "• <b>No ground-truth validation:</b> Findings are descriptive. Clinical decisions should "
        "consider additional context (force measurement, movement observation, patient history).<br/>"
        "• <b>LLM output:</b> Provided as a summarisation aid, not a diagnostic tool. "
        "Human-in-the-loop review is required.",
        STYLES['BodyJustify']))

    story.append(Paragraph(
        "<i>Report generated automatically by the sEMG Interpretation Pipeline. For "
        "clinical use, please corroborate findings with additional assessments.</i>",
        STYLES['SmallNote']))

    return story


# ==============================
# BUILD REPORT
# ==============================

def build_participant_story(data):
    """Build the document story for one participant."""
    participant = data['participant']
    features = data['features']
    result = data['result']

    story = []
    story += build_cover(participant, features, result)

    if features:
        story += build_per_sensor_section(features, result)
        story.append(Spacer(1, 0.3 * cm))
        story += build_cross_channel_section(features, result)
        mu = build_mu_recruitment_section(result)
        if mu:
            story.append(PageBreak())
            story += mu
        viz = build_visualization_section(data['plot_path'])
        if viz:
            story.append(Spacer(1, 0.3 * cm))
            story += viz
        story += build_methodology(features, result)
    return story


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawString(1.5 * cm, 1 * cm,
                      f"sEMG Activation Report  —  Page {canvas.getPageNumber()}")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm,
                           datetime.now().strftime('%Y-%m-%d'))
    canvas.restoreState()


def generate_participant_pdf(participant):
    data = load_data(participant)
    if not data['features']:
        print(f"  SKIP {participant}: no features file")
        return None
    if not data['result']:
        print(f"  SKIP {participant}: no result file")
        return None

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f'{participant}_activation_report.pdf')

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=2*cm,
        title=f"sEMG Muscle Activation Report — {participant}",
        author="sEMG Interpretation Pipeline",
    )
    story = build_participant_story(data)
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  {participant}: {out_path} ({size_kb:.0f} KB)")
    return out_path


def generate_combined_pdf(participants):
    """One combined PDF with all participants."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, 'All_Participants_Report.pdf')

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=2*cm,
        title="sEMG Muscle Activation Report — All Participants",
        author="sEMG Interpretation Pipeline",
    )

    full_story = []

    # Main cover
    full_story.append(Spacer(1, 3 * cm))
    full_story.append(Paragraph("sEMG Muscle Activation Report", STYLES['ClinicalTitle']))
    full_story.append(Paragraph(
        f"All Participants &nbsp;·&nbsp; Gesture_1 &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%B %d, %Y')}",
        STYLES['Subtitle']))
    full_story.append(Spacer(1, 1 * cm))
    full_story.append(Paragraph(
        "This document contains individual muscle-activation reports for each participant. "
        "Each report summarises per-sensor activation, spatial patterns, motor-unit recruitment "
        "indicators, visualisation, and methodology. Comparisons across participants should be "
        "interpreted with caution: without %MVC normalisation, absolute amplitude values are "
        "not directly comparable between subjects.",
        STYLES['BodyJustify']))

    loaded = 0
    for p in participants:
        data = load_data(p)
        if not data['features'] or not data['result']:
            print(f"  SKIP {p} in combined PDF")
            continue
        full_story.append(PageBreak())
        full_story += build_participant_story(data)
        loaded += 1

    if loaded == 0:
        print("No participants had complete data — skipping combined PDF")
        return None

    doc.build(full_story, onFirstPage=add_footer, onLaterPages=add_footer)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nCombined: {out_path} ({size_kb:.0f} KB, {loaded} participants)")
    return out_path


def main():
    if len(sys.argv) > 1:
        participants = sys.argv[1:]
    else:
        # Auto-discover
        participants = []
        for p in PARTICIPANTS:
            if os.path.exists(os.path.join(DATA_DIR, p, 'activation_result.json')):
                participants.append(p)

    if not participants:
        print("No participants with activation_result.json found")
        sys.exit(1)

    print(f"Generating PDFs for: {participants}\n")
    for p in participants:
        generate_participant_pdf(p)

    if len(participants) > 1:
        print("\nBuilding combined report...")
        generate_combined_pdf(participants)

    print("\nDone. PDFs in:", REPORTS_DIR)


if __name__ == '__main__':
    main()
