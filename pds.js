window.onload = function () {
  const image = document.getElementById('displayedImage');
  const picElement = document.getElementById('pic');
  const txtMainElement = document.getElementById('txt_main');
  const predictText = document.getElementById('predictText');
  const fetchImage = async () => {
      try {
          const response = await fetch('http://localhost:8000/image');
          if (response.ok) {
              const blob = await response.blob();
              const imageUrl = URL.createObjectURL(blob);
              image.src = imageUrl;
              image.style.display='block';
              picElement.style.display = 'none';
              txtMainElement.style.display = 'none';
              await fetch('http://localhost:8000/get_text')
            .then(response1 => response1.json())
            .then(data => {
              if(data.c_name=="Not Pneumonic.png"){
                predictText.innerText = "Not Pneumonic";
                var elements = document.getElementById("predictText");
                elements.classList.remove("prediction-text");
                elements.classList.add("prediction-text_not");
              }
              else{
               
                predictText.innerText = "Pneumonic";
              }
            })
            .catch(error => console.error('Error:', error));
              predictText.style.display = 'block';
          } else {
              throw new Error('Error fetching image');
          }
      } catch (error) {
          console.error('Error fetching image:', error);
          // Retry fetching after a delay
          setTimeout(fetchImage, 8000); // Retry after 2 seconds
      }
  };

  // Initial fetch
  fetchImage();
};
document.getElementById('resetButton').addEventListener('click', async () => {
  try {
      const response = await fetch('http://localhost:8000/delete-image', {
          method: 'DELETE',
      });

      console.log(response.status);  // Log the response status

      if (response.ok) {
          console.log('Image deleted successfully');
      } else {
          throw new Error('Error deleting image');
      }
  } catch (error) {
      console.error('Error deleting image:', error);
  }
});


  const image_input = document.querySelector("#image_input");
  var uploaded_image = "";

  image_input.addEventListener("change", function () {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      uploaded_image = reader.result;
      document.querySelector("#img-place1").style.backgroundImage = `url(${uploaded_image})`;
    });
    reader.readAsDataURL(this.files[0]);
    remove();
    remove1();
  });

  function remove() {
    var element = document.getElementById("txt_main");
    element.classList.remove("txt");
    element.classList.add("txt5");
  }

  function remove1() {
    var elements = document.getElementById("pic");
    elements.classList.remove("icon1");
    elements.classList.add("plus-icon5");
  }

  const be = document.getElementById("backend");
  be.addEventListener('click', async function submitImage() {
    const fileInput = document.getElementById('image_input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      const imagePath = data.image_path;
      console.log(data);

      document.getElementById('displayedImage').src = imagePath;
    } catch (error) {
      console.error('Error submitting image:', error);
    }
  });


  window.addEventListener("beforeunload", function(event) {
    try {
      const response = fetch('http://localhost:8000/delete-image', {
          method: 'DELETE',
      });
 
      console.log(response.status);  // Log the response status

      if (response.ok) {
          console.log('Image deleted successfully');
      } else {
          throw new Error('Error deleting image');
      }
  } catch (error) {
      console.error('Error deleting image:', error);
  }
});