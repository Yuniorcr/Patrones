const prediccion = document.getElementById("prediction");

function sendData(sentence){
    fetch('https://cyberbullyng.herokuapp.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify({
            sentence: sentence
        })
    })
    .then(response => response.json())
    .then(data => {
        if(data.data >0.5){
            prediccion.innerHTML = '';
            prediccion.innerHTML = 
            `<div class="alert alert-danger" role="alert">
                <h4 class="alert-heading"><p><mark>${sentence}</mark> </p>bullyng!</h4>
            </div>
            <div class="alert alert-info">
                Score: ${data.data}
            </div>`
        }else{
            prediccion.innerHTML = '';
            prediccion.innerHTML = 
            `<div class="alert alert-success">
                <h4 class="alert-heading"><p><mark>${sentence}</mark> </p>Not bullyng!</h4>
            </div>
            <div class="alert alert-info">
                Score: ${data.data}
            </div>
            `
        }
    }
    )
}

document.getElementById("btn-start").addEventListener("click", () => {
    // clean spaces start and end
    var data = document.getElementById("bullyng");
    if(data.value.trim().length > 0){
        lower = data.value.trim().toLowerCase();
        sendData(lower);
        data.value = '';
        prediccion.innerHTML = `
        <div class="d-flex justify-content-center">
            <div class="spinner-border text-info" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
        `;
    }else{
        prediccion.innerHTML = '';
        prediccion.innerHTML = `
        <div class="alert alert-danger" role="alert">
            Debes escribir algo
        </div>`
    }
});

document.getElementById("bullyng").addEventListener("keyup", (event) => {
    if(event.keyCode === 13){
        var data = document.getElementById("bullyng");
        if(data.value.trim().length > 0){
            lower = data.value.trim().toLowerCase();
            sendData(lower);
            data.value = '';
            prediccion.innerHTML = `
            <div class="d-flex justify-content-center">
                <div class="spinner-border text-info" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
            `;
        }else{
            prediccion.innerHTML = '';
            prediccion.innerHTML = `
            <div class="alert alert-danger" role="alert">
                Debes escribir algo
            </div>`
        }
    }
});
