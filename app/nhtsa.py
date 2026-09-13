import urllib.request
import urllib.parse
import json
from .log import get_logger

logger = get_logger("nhtsa")

def fetch_nhtsa_data(make: str, model: str, year: int) -> tuple[list[dict], list[dict]]:
    """Fetch Recalls and Complaints from NHTSA API for the given make, model, and year.
    Returns (recalls, complaints).
    """
    if not make or not model or not year:
        return [], []
    
    make_clean = urllib.parse.quote(make.strip())
    model_clean = urllib.parse.quote(model.strip())
    
    recalls = []
    try:
        url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make_clean}&model={model_clean}&modelYear={year}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            results = data.get("results", [])
            for r in results:
                recalls.append({
                    "id": r.get("NHTSACampaignNumber", ""),
                    "date": r.get("ReportReceivedDate", ""),
                    "component": r.get("Component", ""),
                    "summary": r.get("Summary", ""),
                    "consequence": r.get("Consequence", "")
                })
    except Exception as e:
        logger.warning(f"Failed to fetch NHTSA recalls for {year} {make} {model}: {e}")

    complaints = []
    try:
        url = f"https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make_clean}&model={model_clean}&modelYear={year}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            results = data.get("results", [])
            for r in results:
                complaints.append({
                    "id": r.get("odiNumber", ""),
                    "date": r.get("dateComplaintFiled", ""),
                    "component": r.get("components", ""),
                    "summary": r.get("summary", ""),
                })
    except Exception as e:
        logger.warning(f"Failed to fetch NHTSA complaints: {e}")
        
    return recalls, complaints
