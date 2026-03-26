from services.analysis_service import transcribe_video

local_file = r"C:\Users\nagal\OneDrive\Documents\GitHub\BoostMatch\BoostMatch\system\TitaSarhie.mp4"
text = transcribe_video(local_file)
print(text)