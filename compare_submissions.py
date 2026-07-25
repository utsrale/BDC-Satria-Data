import pandas as pd
import subprocess
import io

def compare():
    # 1. Load new submission
    new_df = pd.read_csv("submission_SD2026040000187.csv")
    
    # 2. Get old submission from git HEAD
    try:
        git_show = subprocess.run(
            ["git", "show", "HEAD:submission_SD2026040000187.csv"],
            capture_output=True,
            text=True,
            check=True
        )
        old_df = pd.read_csv(io.StringIO(git_show.stdout))
    except Exception as e:
        print("Gagal mengambil versi lama dari Git:", e)
        return
    
    # Check if lengths match
    if len(new_df) != len(old_df):
        print(f"Perhatian: Jumlah baris berbeda! (Lama: {len(old_df)}, Baru: {len(new_df)})")
        return
    
    # Align by 'id'
    new_df = new_df.sort_values(by='id').reset_index(drop=True)
    old_df = old_df.sort_values(by='id').reset_index(drop=True)
    
    # Compare
    diff_mask = new_df['predicted'] != old_df['predicted']
    diff_count = diff_mask.sum()
    total_count = len(new_df)
    percent = (diff_count / total_count) * 100
    
    print("=" * 60)
    print(f"HASIL PERBANDINGAN SUBMISSION")
    print("=" * 60)
    print(f"Total baris data             : {total_count}")
    print(f"Jumlah label yang BERUBAH    : {diff_count} ({percent:.2f}%)")
    print(f"Jumlah label yang SAMA       : {total_count - diff_count} ({100 - percent:.2f}%)")
    print("-" * 60)
    
    # Class map
    idx_to_class = {0: 'Recyclable', 1: 'Electronic', 2: 'Organic'}
    
    # Distribution
    print("Distribusi Kelas Versi LAMA:")
    print(old_df['predicted'].value_counts().rename(index=idx_to_class))
    print("\nDistribusi Kelas Versi BARU (Val F1 0.91):")
    print(new_df['predicted'].value_counts().rename(index=idx_to_class))
    print("-" * 60)
    
    # Changes breakdown
    diff_df = pd.DataFrame({
        'id': old_df.loc[diff_mask, 'id'],
        'lama': old_df.loc[diff_mask, 'predicted'].map(idx_to_class),
        'baru': new_df.loc[diff_mask, 'predicted'].map(idx_to_class)
    })
    
    print("Detail Perubahan Label:")
    changes = diff_df.groupby(['lama', 'baru']).size().reset_index(name='count').sort_values(by='count', ascending=False)
    for _, row in changes.iterrows():
        print(f"  {row['lama']} -> {row['baru']} : {row['count']} gambar")
    print("=" * 60)

if __name__ == "__main__":
    compare()
