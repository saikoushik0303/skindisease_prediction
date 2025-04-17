document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predict-btn").addEventListener("click", uploadImage);
});

// Function to upload image and get prediction
function uploadImage() {
    let fileInput = document.getElementById("file-upload");
    let formData = new FormData();
    
    if (!fileInput.files.length) {
        alert("Please select an image to upload!");
        return;
    }

    formData.append("file", fileInput.files[0]);

    fetch("/upload", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === "File uploaded successfully") {
            document.getElementById("prediction-result").innerHTML = `
                <div class="result-card">
                    <h3>Prediction: ${data.disease}</h3>
                    <p><strong>Medicine:</strong> ${data.medicine}</p>
                    <p><strong>Consult:</strong> ${data.doctor}</p>
                </div>
            `;
        } else {
            alert("Prediction failed! Try again.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Error uploading file!");
    });
}
