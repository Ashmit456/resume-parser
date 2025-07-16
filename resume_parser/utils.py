from typing import Any, Dict, List, Optional
import copy
import re
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import chromadb
import csv

# Initialize SBERT model
print("[DEBUG] Loading SBERT model 'all-mpnet-base-v2'")
model = SentenceTransformer('all-mpnet-base-v2')
print("[DEBUG] SBERT model initialized.")

COMPANY_CSV_FILE = "companies.csv"
SKILL_CSV_FILE = "csv_db_cleaned.csv"
COLLEGE_CSV_FILE = "india_engg_colleges_updated(in).csv"
TITLE_CSV_FILE = "titles.csv"

# ─── Enum Mappings ────────────────────────────────────────────────────────────
# Education enum: maps normalized degree keywords to (code, string_key)
EDUCATION_ENUM = {
    "diploma":       (100, "educationcode.diploma"),
    "polytechnic":   (100, "educationcode.diploma"),
    "bachelor":      (200, "educationcode.bachelors"),
    "bachelors":     (200, "educationcode.bachelors"),
    "btech":         (200, "educationcode.bachelors"),
    "b\.tech":      (200, "educationcode.bachelors"),
    "be":            (200, "educationcode.bachelors"),
    "b\.e":         (200, "educationcode.bachelors"),
    "bsc":           (200, "educationcode.bachelors"),
    "b\.sc":        (200, "educationcode.bachelors"),
    "ba":            (200, "educationcode.bachelors"),
    "master":        (300, "educationcode.masters"),
    "masters":       (300, "educationcode.masters"),
    "mtech":         (300, "educationcode.masters"),
    "m\.tech":      (300, "educationcode.masters"),
    "me":            (300, "educationcode.masters"),
    "m\.e":         (300, "educationcode.masters"),
    "msc":           (300, "educationcode.masters"),
    "m\.sc":        (300, "educationcode.masters"),
    "ma":            (300, "educationcode.masters"),
    "mba":           (300, "educationcode.masters"),
    "doctorate":     (400, "educationcode.doctorate"),
    "phd":           (400, "educationcode.doctorate"),
    "dphil":         (400, "educationcode.doctorate"),
    "md":            (400, "educationcode.doctorate"),
}

# Experience‐level enum
EXPERIENCE_ENUM = {
    "student":         (1, "experiencelevel.student"),
    "fresher":         (2, "experiencelevel.fresher"),
    "experienced":     (3, "experiencelevel.experienced"),

}

# Gender enum
GENDER_ENUM = {
    "male":            (1, "gender.male"),
    "m":               (1, "gender.male"),
    "man":             (1, "gender.male"),
    "boy":             (1, "gender.male"),
    "female":          (2, "gender.female"),
    "f":               (2, "gender.female"),
    "woman":           (2, "gender.female"),
    "girl":            (2, "gender.female"),
    "no_disclose":     (3, "gender.no.disclose"),
    "nodisclose":      (3, "gender.no.disclose"),
    "n/a":             (3, "gender.no.disclose"),
    "other":           (3, "gender.no.disclose"),
    "nonbinary":       (3, "gender.no.disclose"),
    "non-binary":      (3, "gender.no.disclose"),
}

CANDIDATE_CLASS_ENUM = {
    "10":         (1, "schoolclass.10"),
    "10th":       (1, "schoolclass.10"),
    "x":          (1, "schoolclass.10"),
    "tenth":      (1, "schoolclass.10"),
    "12":         (2, "schoolclass.12"),
    "12th":       (2, "schoolclass.12"),
    "xii":        (2, "schoolclass.12"),
    "twelfth":    (2, "schoolclass.12"),
}

def normalize_string(s: Optional[str]) -> str:
    return (s or "").strip().lower().replace("-", " ").replace(".", "").replace(",", "").replace("/", " ").replace("  ", " ")

def normalize_skill(skill: Optional[str]) -> str:
    return (skill or "").strip().lower()

def remove_empty(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: remove_empty(v) for k, v in d.items() if v not in [None, "", [], {}]}
    elif isinstance(d, list):
        return [remove_empty(v) for v in d if v not in [None, "", [], {}]]
    else:
        return d

# ─── Lookup Functions ─────────────────────────────────────────────────────────
def lookup_skill_id(skill: str, k: int = 1) -> List[Dict[str, Any]]:
    print(f"[DEBUG] lookup_skill_id called with: {skill}")
    client = chromadb.PersistentClient(path="chroma_db_api")
    col = client.get_or_create_collection(name="skills", metadata={"hnsw:space": "cosine"})
    q = normalize_skill(skill)
    emb = model.encode([q], normalize_embeddings=True)[0]
    res = col.query(
        query_embeddings=[emb.tolist()],
        n_results=k,
        include=["metadatas", "distances"]
    )
    print(f"[DEBUG] Chromadb skill ids: {res['ids'][0]}")
    return [
        {
            "skill_id": sid,
            "skill": meta.get("skill", ""),
            "distance": dist
        }
        for sid, meta, dist in zip(res['ids'][0], res['metadatas'][0], res['distances'][0])
    ]


def lookup_college_id(college: str, k: int = 1) -> List[Dict[str, Any]]:
    print(f"[DEBUG] lookup_college_id called with: {college}")
    client = chromadb.PersistentClient(path="chroma_db_api")
    col = client.get_or_create_collection(name="colleges", metadata={"hnsw:space": "cosine"})
    q = normalize_string(college)
    emb = model.encode([q], normalize_embeddings=True)[0]
    res = col.query(
        query_embeddings=[emb.tolist()],
        n_results=k,
        include=["metadatas", "distances"]
    )
    return [
        {
            "college_id": cid,
            "college_name": meta.get("college", ""),
            "distance": dist
        }
        for cid, meta, dist in zip(res['ids'][0], res['metadatas'][0], res['distances'][0])
    ]


def lookup_company_id(query: str, k: int = 1) -> List[Dict[str, Any]]:
    print(f"[DEBUG] lookup_company_id called with: {query}")
    client = chromadb.PersistentClient(path="chroma_db_api")
    col = client.get_or_create_collection(name="companies", metadata={"hnsw:space": "cosine"})
    q = normalize_string(query)
    emb = model.encode([q], normalize_embeddings=True)[0]
    res = col.query(
        query_embeddings=[emb.tolist()],
        n_results=k,
        include=["metadatas", "distances"]
    )
    print(f"[DEBUG] Chromadb college ids: {res['ids'][0]}")
    return [
        {
            "company_id": sid,
            "company_name": meta.get("companie", ""),
            "distance": dist
        }
        for sid, meta, dist in zip(res['ids'][0], res['metadatas'][0], res['distances'][0])
    ]

def lookup_title_id(title: str, k: int = 1) -> List[Dict[str, Any]]:
    print(f"[DEBUG] lookup_title_id called with: {title}")
    # Connect to the persistent database
    client = chromadb.PersistentClient(path="chroma_db_api")
    # Ensure the 'titles' collection exists
    col = client.get_or_create_collection(
        name="titles",
        metadata={"hnsw:space": "cosine"}
    )
    # Normalize and embed the query title
    q_norm = normalize_string(title)
    emb = model.encode([q_norm], normalize_embeddings=True)[0]
    # Perform the nearest-neighbor search
    res = col.query(
        query_embeddings=[emb.tolist()],
        n_results=k,
        include=["metadatas", "distances"]
    )
    print(f"[DEBUG] Chromadb title ids: {res['ids'][0]}")
    # Build and return the result list
    results: List[Dict[str, Any]] = []
    for sid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "title_id": sid,
            "title": meta.get("title", ""),
            "distance": dist
        })
    return results

# ─── Enum Finders ─────────────────────────────────────────────────────────────
def find_candidate_class_key(raw_class: str) -> Optional[int]:
    if not raw_class:
        return None
    norm = normalize_string(str(raw_class))
    return CANDIDATE_CLASS_ENUM.get(norm, (None, None))[0]


def find_education_key(raw_degree: str) -> (Optional[int], Optional[str]):
    print(f"[DEBUG] find_education_key called with: {raw_degree}")
    norm = normalize_string(raw_degree)

    for label, (code, key) in EDUCATION_ENUM.items():
        pat = label.replace(" ", "\\s+")
        if re.search(rf"\b{pat}\b", norm):
            return code, key
    return None, None


def find_experience_key(raw_exp: Optional[str]) -> (Optional[int], Optional[str]):
    print(f"[DEBUG] find_experience_key called with: {raw_exp}")    
    norm = normalize_string(raw_exp)
    for label, (code, key) in EXPERIENCE_ENUM.items():
        if re.search(rf"\b{label}\b", norm):
            return code, key
    return None, None


def find_gender_key(raw_gender: Optional[str]) -> (Optional[int], Optional[str]):
    print(f"[DEBUG] find_gender_key called with: {raw_gender}")
    return GENDER_ENUM.get(normalize_string(raw_gender).replace(" ", "_"), (None, None))


def apply_enum_mappings(resume: Dict[str, Any]) -> Dict[str, Any]:
    print("[DEBUG] Applying enum mappings to resume")
    for e in resume.get("education", []):
        e["degree"] = find_education_key(e.get("degree"))[0]
    resume["experienceLevel"] = find_experience_key(resume.get("experienceLevel"))[0]
    resume["gender"] = find_gender_key(resume.get("gender"))[0]
    return resume

# ─── City Mapping Helper ──────────────────────────────────────────────────────
def map_city(src: Any) -> Dict[str, Any]:
    print(f"[DEBUG] map_city called with: {src}")
    city = {"name": None, "shortname": None, "state": None, "placeid": None,
            "location": {"x": None, "y": None, "coordinates": [None, None], "type": None}}
    if not isinstance(src, dict):
        return city
    for k in ("name", "shortname", "state", "placeid"):
        city[k] = copy.deepcopy(src.get(k))
    loc = src.get("location", {})
    if isinstance(loc, dict):
        for sub in ("x", "y", "type"):
            city["location"][sub] = copy.deepcopy(loc.get(sub))
        coords = loc.get("coordinates", [])
        if isinstance(coords, list) and len(coords) >= 2:
            city["location"]["coordinates"] = [coords[0], coords[1]]
    return city

# ─── Main Conversion ──────────────────────────────────────────────────────────
def convert_resume(source: Dict[str, Any]) -> Dict[str, Any]:
    def _c(d: Dict[str, Any], k: str) -> Any:
        return copy.deepcopy(d.get(k))

    print(source)

    # Base structure
    output: Dict[str, Any] = {
        "name": _c(source, "name"),
        "autounapplyself": False,
        "email": _c(source, "email"),
        "mobile": _c(source, "mobile"),
        "state": 2,
        "address": _c(source, "address"),
        "city": map_city(source.get("city")),
        "anyjoblocation": True,
        "jobtype": 300,
        "gender": None,
        "loctype": 1,
        "expSalary": {"any": False, "salary": None, "negotiable": False},
        "experienceLevel": _c(source, "experienceLevel"),
        "jobexp": _c(source, "jobexp"),
        "relocate": True,
        "joining": 200,
        "linkedin": _c(source, "linkedin"),
        "twitter": _c(source, "twitter"),
        "instagram": _c(source, "instagram"),
        "facebook": _c(source, "facebook"),
        "education": [],
        "schooling": [],
        "breaks": [],
        "jobs": [],
        "skills": [],
        "tools": [],
        "certifications": [],
        "achievements": _c(source, "achievements"),
        "acadexcellence": _c(source, "acadexcellence"),
        "patents": _c(source, "patents"),
        "projects": _c(source, "projects"),
        "hobbies": _c(source, "hobbies"),
        "dob": "",
        "about": _c(source, "about"),
    }


    # Load CSV mappings
    id_to_skill: Dict[str, str] = {}
    with open(SKILL_CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_skill[row["s_no"]] = row["name"]

    id_to_college: Dict[str, str] = {}
    with open(COLLEGE_CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_college[row["CollegeID"]] = row["College Name"]
    print("getting colleges")
    
    # Process education entries
    for edu in source.get("education", []):
        college_raw = _c(edu, "college") or ""
        if college_raw:
            matches = lookup_college_id(college_raw, k=1)
            if matches and matches[0]["distance"] <= 0.30:
                cid = matches[0]["college_id"]
                college = id_to_college.get(cid, matches[0]["college_name"])
            else:
                print(f"Warning: No close match found for college '{college_raw}'")
                college = college_raw
                cid = -1  # Use -1 to indicate no match

        # New: pull pct and cgpa
        pct_raw = _c(edu, "pct")
        cgpa_raw = _c(edu, "cgpa")

        # Decide scoreType
        if pct_raw is not None:
            score_type = 1
        elif cgpa_raw is not None:
            score_type = 2
        else:
            score_type = 1

        ne = {
            "college": college,
            "collegeId": cid,
            "degree": _c(edu, "degree"),
            "from": _c(edu, "from"),
            "to": _c(edu, "to"),
            "scoreType": score_type,
            "pct": pct_raw,
            "cgpa": cgpa_raw,
            "city": map_city(edu.get("city")),
        }
        output["education"].append(ne)


    # Process schooling entries
    print("getting schooling")
    for sch in source.get("schooling", []):
        mapped_class = find_candidate_class_key(_c(sch, "candidateClass"))
        ns = {
            "name": _c(sch, "name"),
            "candidateClass": mapped_class,
            "board": _c(sch, "board"),
            "pct": _c(sch, "pct"),
            "city": map_city(sch.get("city")),
        }
        output["schooling"].append(ns)

    # Process job entries

    id_to_company: Dict[str, str] = {}
    with open(COMPANY_CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_company[row["Company ID"]] = row["Company"]

    id_to_title: Dict[str, str] = {}
    with open(TITLE_CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_title[row["Job Title Id"]] = row["Name"]

    print("getting jobs")
    for job in source.get("jobs") or []:
        raw_title = _c(job, "title") or ""
        raw_company = _c(job, "company") or ""

        # Title match
        matches_title = lookup_title_id(raw_title, k=1)
        if matches_title and matches_title[0]["distance"] <= 0.30:
            title_id = matches_title[0]["title_id"]
            title_name = id_to_title.get(title_id, matches_title[0]["title"])
        else:
            title_id = -1
            title_name = raw_title

        # Company match
        matches_company = lookup_company_id(raw_company, k=1)
        if matches_company and matches_company[0]["distance"] <= 0.30:
            company_id = matches_company[0]["company_id"]
            company_name = id_to_company.get(company_id, matches_company[0]["company_name"])
        else:
            company_id = -1
            company_name = raw_company

        mapped_job = {
            "title": title_name,
            "titleid": title_id,
            "company": company_name,
            "companyId": company_id,
            "from": _c(job, "from"),
            "to": _c(job, "to"),
            "city": map_city(_c(job, "city") or ""),
        }
        output["jobs"].append(mapped_job)

    # Compute derived experience parameters
    print("computing derived experience")
    exp_level = find_experience_key(output.get("experienceLevel"))[0]
    jobexp_years = output.get("jobexp") or 0
    derived_exp = 3 if exp_level in [1, 2] else jobexp_years * 12
    derived_where = 3 if exp_level in [1, 2] else 1

    print("getting skills")
    for sk in source.get("skills", []):
        nm = _c(sk, "name") or ""
        matches = lookup_skill_id(nm, k=1)
        if matches and matches[0]["distance"] <= 0.30:
            sid = matches[0]["skill_id"]
            print("id:", sid, "skill:", matches[0]["skill"])
            canon = id_to_skill.get(sid, matches[0]["skill"]) 
            print("canon:", canon)
            output["skills"].append({
                "name": canon,
                "id": sid,
                "experience": derived_exp,
                "rating": 200,
                "where": derived_where,
            })

    # Process tools entries
    print("getting tools")
    for tl in source.get("tools", []):
        output["tools"].append({
            "name": _c(tl, "name"),
            "experience": derived_exp,
            "rating": 200,
            "lastused": _c(tl, "lastused"),
            "where": derived_where,
        })

    # Process certifications entries
    print("getting certifications")
    for cert in source.get("certifications") or []:
        output["certifications"].append({"name": _c(cert, "name"), "certificateid": None})

    # Format DOB
    print("getting dob")
    if isinstance(source.get("dob"), str):
        dob_raw = source.get("dob") or ""
        output["dob"] = dob_raw + "T00:00:00+05:30" if len(dob_raw) == 10 else dob_raw

    # Build skillsStr if any
    print("getting skillsStr")
    if output["skills"]:
        output["skillsStr"] = ", ".join(s["name"] for s in output["skills"])

    # Adjust jobexp for student/fresher
    print("getting jobexp")
    if exp_level in [1, 2]:
        output["jobexp"] = -1

    # Apply final enum mappings
    print("getting final mappings")
    output = apply_enum_mappings(output)
    
    # Remove empty, null, or None values
    print("getting final output")
    output = remove_empty(output)
    
    return output
