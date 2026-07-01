import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load model once and cache it to keep the app fast
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 1. Dashboard Headers
st.title("Brain Scan Analysis Dashboard")
st.subheader("Automated Tumor Detection and Identification System")
st.divider()

# 2. File Uploader Box (Drag and Drop)
uploaded_file = st.file_uploader(
    label="Upload Brain Scan or X-Ray Image", 
    type=["jpg", "jpeg", "png"]
)

# 3. Handle File Actions Dynamically
if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    
    # Run live YOLO inference on the image
    with st.spinner("Analyzing image with AI model..."):
        results = model.predict(image, conf=0.25) # Returns a list of Results
        
    # Get the first result object
    result = results[0]
    
    # Plot the bounding boxes directly onto the image
    # result.plot() returns a numpy array (BGR format), which PIL converts easily
    annotated_image_array = result.plot()
    annotated_image = Image.fromarray(annotated_image_array[..., ::-1]) # Convert BGR to RGB
    
    # Display the final analyzed image with boxes
    st.image(annotated_image, caption="Analyzed Scan Image", use_container_width=True)
    st.success("Analysis complete!")
    st.divider()
    
    # Check if any tumors were actually detected
    if len(result.boxes) > 0:
        # Extract data from the first detected box for the report string
        top_box = result.boxes[0]
        class_id = int(top_box.cls[0].item())
        tumor_name = result.names[class_id] # E.g., 'meningioma', 'glioma', 'pituitary'
        confidence_score = float(top_box.conf[0].item()) * 100 # Convert to percentage
        
        # 4. Main Dynamic Patient-Friendly Content
        patient_report = f"""
        ### What this means for you:
        * **The Scan is Processed**: Our AI system has successfully checked the uploaded brain scan.
        * **Tumor Identified**: The system located **1 {tumor_name}** on the scan image. 
        * **High Certainty**: The software is **{confidence_score:.1f}% confident** in this specific identification. 

        ---

        ### Key Information on Tumor Types:
        To help you understand the different categories our system checks for, here is a quick breakdown of what they mean:
        * **Meningioma**: A tumor that grows on the outer protective membranes surrounding the brain. They are usually slow-growing and have clear, distinct borders on a scan.
        * **Glioma**: A tumor that begins inside the brain tissue itself. These can have less defined edges and often blend into the surrounding areas.
        * **Pituitary**: A tumor that grows specifically on the pituitary gland at the base of the skull, which controls the body's hormones.
        """
        st.markdown(patient_report)
    else:
        # Display this if the model runs but finds no boxes
        st.balloons()
        st.info("### What this means for you:\n* **No tumors detected**: The AI system scanned the image and did not identify signs of a glioma, meningioma, or pituitary tumor.")
    
    st.divider()

else:
    # Message shown when the user hasn't uploaded a file yet
    st.info("Please drop a brain scan or X-ray image above to begin the automated analysis.")

# 5. Fixed Disclaimer Box at the Bottom
st.caption(
    "*Disclaimer: This software is designed as an analytical tool to support image review. "
    "It does not provide an official medical diagnosis. All automated findings should be reviewed "
    "and verified by a qualified radiologist or physician.*"
)
