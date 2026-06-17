def segment_transcript(transcription_result: dict) -> list:
    """
    Break down the transcription into clauses/sentences.
    Uses basic punctuation and pauses.
    """
    # A simple initial approach: Combine sentences provided by Whisper directly.
    # Whisper usually segments text quite cleanly out of the box.
    
    segments = transcription_result.get("segments", [])
    
    sentences = []
    for s in segments:
        text = s.get("text", "").strip()
        if text:
            sentences.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(text)
            })
            
    return sentences
