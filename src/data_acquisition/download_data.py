import wfdb
import os

def download_ptb_xl_subset(output_dir, records_to_download=50):
    """
    Downloads a subset of the PTB-XL dataset using wfdb.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading PTB-XL subset to {output_dir}...")
    
    try:
        # Download the RECORDS file first
        wfdb.dl_files('ptb-xl', output_dir, ['RECORDS'])
        
        with open(os.path.join(output_dir, 'RECORDS'), 'r') as f:
            records = [line.strip() for line in f.readlines()]
            
        # Limit to subset
        subset = records[:records_to_download]
        print(f"Found {len(records)} records. Downloading first {len(subset)}...")
        
        for record in subset:
            # Check if files already exist to avoid re-downloading
            base_name = record
            # PTB-XL records are in subfolders like records100/00000/00001_hr
            # wfdb.dl_files handles the directory structure
            
            # We need .dat and .hea files. 
            # Note: PTB-XL has _hr (high res) and standard. We'll try standard first.
            
            try:
                wfdb.dl_files('ptb-xl', output_dir, [record + '.dat', record + '.hea'])
                print(f"Downloaded: {record}")
            except Exception as e:
                print(f"Failed to download {record}: {e}")
            
        print("Download complete.")
        
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    
    download_ptb_xl_subset(raw_data_dir, records_to_download=50) # Download 50 records
