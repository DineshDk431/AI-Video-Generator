"""
Cloud Worker for Video Generation.
Uses Alibaba WAN 2.2 for high-quality video output.
"""
import time
import os
from utils.firebase_utils import init_firebase
from firebase_admin import firestore


def get_generator():
    """Get the video generator."""
    from models.modelscope import get_modelscope_generator
    return get_modelscope_generator(), "wan_2.2"


def process_jobs():
    """Main job processing loop."""
    print("🚀 Cloud Worker Started... Waiting for jobs...")
    print("📦 Loading Alibaba WAN 2.2 video model...")
    
    # Get generator
    generator, model_name = get_generator()
    print(f"✅ Using model: {model_name}")
    
    # Load model
    generator.load_model(progress_callback=lambda msg: print(f"   {msg}"))
    
    # Initialize Firebase
    db = init_firebase()
    if not db:
        print("❌ Firebase Init Failed")
        return

    print("🔄 Polling for jobs...")
    
    while True:
        try:
            # Query for pending jobs
            jobs_ref = db.collection("video_queue").where("status", "==", "pending").limit(1)
            docs = jobs_ref.get()
            
            for doc in docs:
                job_id = doc.id
                data = doc.to_dict()
                print(f"\n📥 Processing Job: {job_id}")
                print(f"📝 Prompt: {data.get('prompt')}")
                
                try:
                    # Mark as processing
                    db.collection("video_queue").document(job_id).update({
                        "status": "processing",
                        "model": model_name
                    })
                    
                    # Get settings
                    settings = data.get("settings", {})
                    prompt = data.get("prompt")
                    video_style = settings.get("video_style", "Cinematic")
                    
                    # Apply style to prompt
                    style_prompts = {
                        "Cinematic": "cinematic, film quality, dramatic lighting, ",
                        "Anime": "anime style, vibrant colors, japanese animation, ",
                        "Normal": ""
                    }
                    enhanced_prompt = style_prompts.get(video_style, "") + prompt
                    
                    # Settings
                    num_frames = settings.get("num_frames", 32)
                    fps = settings.get("fps", 8)
                    height = settings.get("height", 512)
                    width = settings.get("width", 512)
                    num_steps = settings.get("num_steps", 40)
                    
                    print(f"🎬 Generating {num_frames} frames at {width}x{height}...")
                    print(f"   Style: {video_style} | Steps: {num_steps}")
                    
                    # Common negative prompt
                    negative_prompt = "blurry, low quality, distorted, pixelated, ugly, bad anatomy, deformed, noisy, grainy, watermark, text"
                    
                    # Enhance prompt
                    quality_terms = "high quality, 4k, detailed, photorealistic"
                    if "quality" not in enhanced_prompt.lower():
                        enhanced_prompt = f"{quality_terms}, {enhanced_prompt}"
                    
                    # Generate video
                    frames = generator.generate(
                        prompt=enhanced_prompt,
                        num_frames=num_frames,
                        num_inference_steps=num_steps,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt,
                        enhance_prompt=True,
                        progress_callback=lambda msg: print(f"   {msg}")
                    )
                    
                    # Save video
                    output_dir = "cloud_outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{job_id}.mp4"
                    output_path = os.path.join(output_dir, filename)
                    
                    generator.save_video(frames, output_path, fps=fps)
                    
                    abs_path = os.path.abspath(output_path)
                    
                    # Update Firestore
                    db.collection("video_queue").document(job_id).update({
                        "status": "completed",
                        "video_url": abs_path,
                        "resolution": f"{width}x{height}",
                        "model_used": model_name,
                        "completed_at": firestore.SERVER_TIMESTAMP
                    })
                    
                    print(f"✅ Job {job_id} Completed!")
                    print(f"   📁 Saved to: {abs_path}")
                    
                except Exception as e:
                    print(f"❌ Job Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    db.collection("video_queue").document(job_id).update({
                        "status": "error",
                        "error": str(e)
                    })
            
            time.sleep(5)
            
        except Exception as e:
            print(f"⚠️ Loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    process_jobs()
