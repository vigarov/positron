import numpy as np

def generate_csv_data(roi, source):
    """Generate CSV data for a specific ROI"""
    height, width = roi.shape[:2]
    
    # Create header
    header = "X,Y,R,G,B"
    rows = [header]
    
    # Add pixel data
    for y in range(height):
        for x in range(width):
            pixel = roi[y, x]
            r, g, b = map(int, pixel)
            rows.append(f"{x},{y},{r},{g},{b}")
    
    # Join into a single string
    return "\n".join(rows)

def calculate_statistics(pixel_data):
    """Calculate various statistics for pixel data"""
    stats = {}
    
    for source, roi in pixel_data.items():
        # Calculate per-channel stats
        channel_stats = []
        
        for c in range(3):  # RGB channels
            channel = roi[:,:,c]
            
            # Basic statistics
            channel_min = np.min(channel)
            channel_max = np.max(channel)
            channel_mean = np.mean(channel)
            channel_median = np.median(channel)
            channel_std = np.std(channel)
            
            channel_stats.append({
                'min': channel_min,
                'max': channel_max,
                'mean': channel_mean,
                'median': channel_median,
                'std': channel_std
            })
        
        # Store per-source stats
        stats[source] = {
            'channels': {
                'R': channel_stats[0],
                'G': channel_stats[1],
                'B': channel_stats[2]
            },
            'dimensions': {
                'height': roi.shape[0],
                'width': roi.shape[1]
            }
        }
    
    return stats

def format_filename(filename):
    """Make a filename safe for most operating systems"""
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove problematic characters
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']:
        filename = filename.replace(char, '')
    
    return filename 