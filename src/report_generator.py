
from fpdf import FPDF
import uuid
from datetime import datetime

class HospitalPDF(FPDF):
    def header(self):

        self.set_fill_color(10, 36, 99)
        self.rect(0, 0, 210, 30, "F")
        self.set_text_color(255, 255, 255)
        self.set_font(
            "Arial",
            "B",
            16
        )

        self.cell(
            0,
            10,
            "CITY GENERAL HOSPITAL",
            ln=True,
            align="C"
        )

        self.set_font(
            "Arial",
            "",
            10
        )

        self.cell(
            0,
            8,
            "AI Assisted Diagnostic Report",
            ln=True,
            align="C"
        )

        self.ln(10)

    def footer(self):

        self.set_y(-15)

        self.set_text_color(
            120,
            120,
            120
        )

        self.set_font(
            "Arial",
            "I",
            8
        )

        self.cell(
            0,
            10,
            f"AI System | Page {self.page_no()}",
            align="C"
        )


def color(level):
    level = level.lower()

    if "high" in level:
        return (255, 200, 200)

    elif "medium" in level:
        return (255, 235, 180)

    else:
        return (200, 255, 200)


def clean(text):
    return str(text).encode(
        "latin-1",
        "ignore"
    ).decode("latin-1")


def generate_report(
    patient_name,
    age,
    disease,
    symptoms,
    severity_level,
    severity_score,
    precautions,
    description,
    shap_result,
    medical_conditions,
    medical_history,
    top_predictions
):

    pdf = HospitalPDF()
    pdf.add_page()
    report_id = str(
        uuid.uuid4()
    )[:8].upper()

    date_now = datetime.now().strftime("%d-%m-%Y")
    time_now = datetime.now().strftime("%I:%M %p")

    pdf.set_text_color(0, 0, 0)

    # PATIENT SUMMARY 

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "PATIENT MEDICAL SUMMARY",
        ln=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        clean(f"Patient Name: {patient_name}"),
        ln=True
    )

    pdf.cell(
        0,
        8,
        clean(f"Age: {age}"),
        ln=True
    )

    pdf.cell(
        0,
        8,
        clean(f"Report ID: {report_id}"),
        ln=True
    )

    pdf.cell(0,8,clean(f"Date: {date_now}"), ln = True)
    pdf.cell(0,8,clean(f"Time : {time_now}"), ln=True)

    pdf.ln(4)

    #  MEDICAL CONDITIONS 

    pdf.set_fill_color(235,245,255)

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "EXISTING MEDICAL CONDITIONS",
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    if medical_conditions:

        pdf.multi_cell(
            0,
            8,
            clean(
                ", ".join(
                    medical_conditions
                )
            )
        )

    else:

        pdf.multi_cell(
            0,
            8,
            "No medical conditions selected."
        )

    pdf.ln(2)

    # MEDICAL HISTORY 

    pdf.set_fill_color(
        245,
        245,
        245
    )

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "PATIENT MEDICAL HISTORY",
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    if medical_history.strip():

        pdf.multi_cell(
            0,
            8,
            clean(medical_history)
        )

    else:

        pdf.multi_cell(
            0,
            8,
            "No medical history provided."
        )

    pdf.ln(3)

    #  SYMPTOMS 

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "OBSERVED SYMPTOMS",
        ln=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.multi_cell(
        0,
        8,
        clean(
            ", ".join(
                map(str, symptoms)
            )
            if symptoms
            else "None"
        )
    )

    pdf.ln(3)

    #  DIAGNOSIS 

    pdf.set_fill_color(
        240,
        240,
        240
    )

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "PRIMARY DIAGNOSIS",
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "B",
        11
    )

    pdf.cell(
        0,
        8,
        clean(f"Disease: {disease}"),
        ln=True
    )

    pdf.set_fill_color(
        *color(severity_level)
    )

    pdf.cell(
        0,
        8,
        clean(
            f"Severity: "
            f"{severity_level} "
            f"({severity_score}/10)"
        ),
        ln=True,
        fill=True
    )

    pdf.ln(4)

    #  TOP PREDICTIONS 

    if top_predictions:

        pdf.set_fill_color(
            225,
            240,
            255
        )

        pdf.set_font(
            "Arial",
            "B",
            12
        )

        pdf.cell(
            0,
            10,
            "TOP PREDICTED DISEASES",
            ln=True,
            fill=True
        )

        pdf.set_font(
            "Arial",
            "",
            11
        )

        for disease_name, probability in top_predictions:

            pdf.cell(
                0,
                8,
                clean(
                    f"{disease_name} : {probability:.2f}%"
                ),
                ln=True
            )

        pdf.ln(3)

    #  DESCRIPTION 

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "MEDICAL DESCRIPTION",
        ln=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.multi_cell(
        0,
        8,
        clean(
            description
            if description
            else
            "No description available"
        )
    )

    pdf.ln(3)

    # WHY WAS THIS DISEASE PREDICTED ? (SHAP)

    pdf.set_fill_color(230,245,255)
    pdf.set_font(
        "Arial",
        "B",
        12
    )
    pdf.cell(
        0,
        10,
        "WHY WAS THIS DISEASE PREDICTED ?",
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    if not shap_result.empty:
        for _, row in shap_result.iterrows():
            pdf.cell(
                0, 8 , clean(f"{row['Symptom']} : {row['Importance']:.4f}"), ln = True)
            
    else:
            pdf.cell(0,8 , "SHAP explanation unavailable.", ln = True)
            pdf.ln(3)

    #  PRECAUTIONS 

    pdf.set_fill_color(
        255,
        245,
        204
    )

    pdf.set_font(
        "Arial",
        "B",
        12
    )

    pdf.cell(
        0,
        10,
        "RECOMMENDED PRECAUTIONS",
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    for p in precautions:

        pdf.cell(
            0,
            8,
            clean(f"• {p}"),
            ln=True
        )

    pdf.ln(5)

    #  DISCLAIMER 

    pdf.set_fill_color(
        255,
        230,
        230
    )

    pdf.set_font(
        "Arial",
        "B",
        10
    )

    pdf.multi_cell(
        0,
        8,
        "DISCLAIMER: This AI-generated report is for educational purposes only. Please consult a qualified doctor for confirmation.",
        fill=True
    )

    file_path = (
        f"hospital_report_"
        f"{report_id}.pdf"
    )

    pdf.output(file_path)
    return file_path