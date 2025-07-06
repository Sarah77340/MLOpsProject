const uploadForm = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');

// Onglets
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.getAttribute('data-tab');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(tab).classList.add('active');
  });
});

// Webcam init
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Erreur caméra :", err));
}

// Upload image preview
uploadForm.file.addEventListener('change', e => {
  const file = e.target.files[0];
  const preview = document.getElementById('preview');
  preview.innerHTML = '';
  if (file) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    preview.appendChild(img);
  }
});

// Envoi image fichier
uploadForm.addEventListener('submit', async e => {
  e.preventDefault();
  const formData = new FormData(uploadForm);
  resultDiv.innerHTML = "Analyse en cours...";
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    afficherResultat(data);
  } catch (err) {
    resultDiv.innerHTML = `<p style="color:red;">Erreur : ${err.message}</p>`;
  }
});

// Capture webcam
captureBtn.addEventListener('click', async () => {
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
  const formData = new FormData();
  formData.append("file", blob, "webcam.jpg");
  resultDiv.innerHTML = "Analyse en cours...";
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    afficherResultat(data);
  } catch (err) {
    resultDiv.innerHTML = `<p style="color:red;">Erreur : ${err.message}</p>`;
  }
});

// URL image
document.getElementById('analyzeUrlBtn').addEventListener('click', async () => {
  const url = document.getElementById('imageUrl').value;
  const preview = document.getElementById('urlPreview');
  preview.innerHTML = `<img src="${url}" />`;
  resultDiv.innerHTML = "Analyse en cours...";
  try {
    const res = await fetch(`/predict_url?image_url=${encodeURIComponent(url)}`);
    const data = await res.json();
    afficherResultat(data);
  } catch (err) {
    resultDiv.innerHTML = `<p style="color:red;">Erreur : ${err.message}</p>`;
  }
});

function afficherResultat(data) {
  if (data.error) {
    resultDiv.innerHTML = `<p style="color:red;">Erreur : ${data.error}</p>`;
  } else if (!data.predictions.length) {
    resultDiv.innerHTML = "Aucun visage détecté.";
  } else {
    resultDiv.innerHTML = "<h3>Résultats :</h3>";
    data.predictions.forEach(pred => {
      resultDiv.innerHTML += `<p><strong>${pred.emotion}</strong> (confiance : ${pred.confidence})</p>`;
    });
  }
}
