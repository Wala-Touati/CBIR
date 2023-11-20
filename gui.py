import streamlit as st
import query
import subprocess
import tempfile
import os


# Streamlit UI
st.title("Content-Based Image Retrieval App")

# Upload query image
query_image = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

# Select similarity calculation method
similarity_method = st.selectbox("Select Similarity Calculation Method", ["color", "daisy", "edge", "gabor", "hog", "vgg", "resnet","similarity-matrix"])

if query_image is not None:
# Save the uploaded image to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "query_image.jpg")  # Save as JPG for simplicity
    query_image.seek(0)  # Move the file pointer to the beginning
    with open(temp_image_path, "wb") as temp_file:
        temp_file.write(query_image.read())

    # Display the uploaded image
    st.image(query_image, caption="Uploaded Query Image", use_column_width=True)

    if st.button("Search"):
        st.text("Looking For Similar Images...")

    # Run query.py as a subprocess
    try:
        #results = subprocess.check_output(["python", "query.py", similarity_method, temp_image_path], universal_newlines=True)
        results = subprocess.check_output(["python", "query.py", similarity_method], universal_newlines=True)
        st.text(results)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running query.py: {e}")

    # Clean up temporary files
    os.remove(temp_image_path)
    os.rmdir(temp_dir)
    '''
    st.write("Top 5 Similar Images:")
    for result in results:
        st.image(result['cls'], caption=f"Similarity: {result['dis']:.2f}", use_column_width=True)
    '''