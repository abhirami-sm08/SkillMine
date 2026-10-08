import sys
import os
import random
import pandas as pd
from datetime import datetime, timedelta

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_config import init_db, SessionLocal
from database.models import ScannedOffer, BlacklistEntity, AnalyticsLog

GENUINE_COMPANIES = [
    ("TechCorp Innovations", "techcorp.com", "careers@techcorp.com"),
    ("Global Systems India", "globalsystems.io", "recruitment@globalsystems.io"),
    ("Infosolutions Ltd", "infosolutions.co.in", "hr@infosolutions.co.in"),
    ("Apex Robotics Lab", "apexrobotics.org", "jobs@apexrobotics.org"),
    ("BioGen Diagnostics", "biogen.com", "talent@biogen.com"),
    ("Constructive Infra", "constructiveinfra.in", "careers@constructiveinfra.in"),
    ("Vanguard Financials", "vanguardfin.com", "campus@vanguardfin.com")
]

SUSPICIOUS_COMPANIES = [
    ("Quick Hire HR Services", "quickhire-jobs.wixsite.com", "quickhire2026@gmail.com"),
    ("FastTrack Interns", "fasttrack-careers.info", "fasttrack_recruiter@yahoo.com"),
    ("Apex Freelance Network", "apex-freelance.xyz", "apexjobs@hotmail.com"),
    ("Global Enterprise Sol", "global-enterprise-hr.blogspot.com", "hr_global_sol@gmail.com")
]

FRAUDULENT_COMPANIES = [
    ("EarnFromHome Ltd", "pay-registration-jobs.biz", "scam-recruiter@gmail.com"),
    ("Crypto Typing Work", "crypto-daily-payout.site", "telegram_job_admin@gmail.com"),
    ("Guaranteed Offer Global", "direct-selection-jobs.top", "pay_laptop_fee@yahoo.com"),
    ("WhatsApp Direct Hiring", "wa-jobs.online", "wa_recruiter_2026@gmail.com")
]

DEPARTMENTS = ["CS/IT", "Electronics & Comm", "Mechanical", "Civil", "Biotechnology", "Business & Finance"]

GENUINE_DESCRIPTIONS = [
    "We are seeking a motivated Software Engineering Intern to join our cloud platform team. Responsibilities include building RESTful APIs using Python, writing unit tests, and collaborating in agile sprints. Requirements: B.Tech in CS/IT or related field, knowledge of Git, Python, and SQL. Standard 6-month internship with performance-based full-time offer conversion.",
    "Junior Embedded Systems Engineer position. Candidate will assist in micro-controller programming (C/C++), PCB testing, and hardware debugging. Candidates must have completed coursework in Digital Signal Processing and Embedded C. Official campus recruitment process with online coding assessment and technical interviews.",
    "Data Analyst Intern wanted for financial reporting project. Skills required: Advanced Excel, SQL, Python (Pandas), and PowerBI. Selected candidate will work closely with senior analysts to build business dashboards. 3 rounds of technical evaluation required.",
    "Biotech Research Assistant role in molecular diagnostics. Responsibilities include handling PCR samples, maintaining laboratory compliance logbooks, and analyzing DNA sequencing data. Required: M.Sc or B.Tech in BioTechnology. No fees or deposits required."
]

SUSPICIOUS_DESCRIPTIONS = [
    "Urgent hiring for Data Entry & Web Research Assistant! Earn up to Rs 45,000 per month. Work only 2 hours daily from mobile or laptop. Immediate selection without tough technical interview. Candidates may be asked to pay a small refundable documentation charge of Rs 850 after receiving initial selection letter. DM recruiter on WhatsApp only.",
    "Part Time Graphic Design & Social Media Intern needed. High daily stipend offered. Immediate joiner required within 24 hours. Contact hiring manager via personal Telegram group link. Training fee of Rs 1,500 applicable for software tools module.",
    "Spot Offer! Mechanical Quality Control Assistant. Direct selection based on resume screening. No experience needed. Candidate must deposit security fee for safety kit and uniform before joining date."
]

FRAUDULENT_DESCRIPTIONS = [
    "CONGRATULATION! DIRECT SELECTION OFFER LETTER ISSUED! Earn Rs 65,000 per month typing captchas and filling online forms from home. NO INTERVIEW REQUIRED! GUARANTEED SELECTION! To claim your employee ID and company laptop, pay mandatory laptop security deposit fee of Rs 4,999 to UPI ID fastpay@upi today itself! Apply now on WhatsApp Wa.me/919876543210!",
    "URGENT HIRING: Python Developer Intern - Immediate Joiner! Work from home, 0 experience required. Earn 50,000 stipend. REGISTRATION FEE of Rs 2,500 must be transferred prior to receiving official agreement copy. Direct Telegram contact @scam_recruiter_admin. PAY REGISTRATION FEE TODAY TO BOOK YOUR SEAT!",
    "100% Guaranteed Placement Drive for All Engineering Branches! Direct selection without any coding interview or exam. Candidates must transfer processing fee of Rs 3,500 via GooglePay / PhonePe to clear HR verification. Laptop and joining kit delivered after payment."
]


def generate_synthetic_dataset(num_records: int = 450) -> pd.DataFrame:
    """Generates synthetic job offer records with realistic text, stipends, and fraud indicators."""
    records = []
    now = datetime.now()

    for i in range(num_records):
        # 50% Genuine, 25% Suspicious, 25% Fraudulent
        rnd = random.random()
        if rnd < 0.50:
            label = "Genuine"
            comp_name, domain, email = random.choice(GENUINE_COMPANIES)
            desc = random.choice(GENUINE_DESCRIPTIONS)
            stipend = float(random.randint(15000, 45000))
            exp_years = float(random.choice([0, 0.5, 1.0, 2.0]))
        elif rnd < 0.75:
            label = "Suspicious"
            comp_name, domain, email = random.choice(SUSPICIOUS_COMPANIES)
            desc = random.choice(SUSPICIOUS_DESCRIPTIONS)
            stipend = float(random.randint(30000, 60000)) # Unusually high for part time
            exp_years = float(random.choice([0, 0.5]))
        else:
            label = "Fraudulent"
            comp_name, domain, email = random.choice(FRAUDULENT_COMPANIES)
            desc = random.choice(FRAUDULENT_DESCRIPTIONS)
            stipend = float(random.choice([50000, 75000, 100000])) # High stipend anomaly
            exp_years = 0.0

        dept = random.choice(DEPARTMENTS)
        title_prefix = random.choice(["Intern", "Junior Developer", "Assistant", "Trainee", "Specialist", "Engineer"])
        title = f"{dept} {title_prefix}"
        posting_date = now - timedelta(days=random.randint(0, 90))

        records.append({
            "title": title,
            "company_name": comp_name,
            "company_domain": domain,
            "contact_email": email,
            "contact_phone": f"+91-{random.randint(7000000000, 9999999999)}",
            "department": dept,
            "location": random.choice(["Remote", "Kochi", "Bangalore", "Trivandrum", "Chennai", "Hybrid"]),
            "stipend": stipend,
            "exp_years": exp_years,
            "description": desc,
            "label": label,
            "source": "synthetic",
            "posting_date": posting_date
        })

    return pd.DataFrame(records)


def seed_database():
    """Initializes SQLite database and seeds initial dataset and blacklists."""
    print("[*] Initializing SQLite Database...")
    init_db()
    session = SessionLocal()

    # Clear existing data if any
    session.query(ScannedOffer).delete()
    session.query(BlacklistEntity).delete()
    session.query(AnalyticsLog).delete()
    session.commit()

    print("[*] Seeding Blacklisted Entities...")
    blacklists = [
        BlacklistEntity(entity_type="DOMAIN", entity_value="pay-registration-jobs.biz", reason="Known registration fee scam domain"),
        BlacklistEntity(entity_type="DOMAIN", entity_value="crypto-daily-payout.site", reason="Phishing and crypto wallet scam"),
        BlacklistEntity(entity_type="DOMAIN", entity_value="direct-selection-jobs.top", reason="Unverified job portal requiring laptop deposit"),
        BlacklistEntity(entity_type="EMAIL", entity_value="scam-recruiter@gmail.com", reason="Reported by 15+ students for payment extortion"),
        BlacklistEntity(entity_type="EMAIL", entity_value="wa_recruiter_2026@gmail.com", reason="Fake hiring coordinator using free gmail"),
        BlacklistEntity(entity_type="COMPANY", entity_value="EarnFromHome Ltd", reason="Fake company entity soliciting money"),
        BlacklistEntity(entity_type="PHONE", entity_value="+91-9876543210", reason="Flagged phone number used in WhatsApp job phishing"),
        BlacklistEntity(entity_type="PAYMENT_HANDLE", entity_value="fastpay@upi", reason="Fraudulent UPI ID associated with laptop deposit scam")
    ]
    session.add_all(blacklists)
    session.commit()

    print("[*] Generating Synthetic Job Offers Dataset...")
    df = generate_synthetic_dataset(num_records=450)
    
    # Save seed dataframe to CSV for model training pipeline
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "synthetic_job_offers.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Synthetic dataset saved to {csv_path} ({len(df)} records).")

    session.close()

if __name__ == "__main__":
    seed_database()
