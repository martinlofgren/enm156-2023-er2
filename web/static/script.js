'use strict'

function imgSubmit(event) {
    const btn = document.getElementById('submit_btn');
    btn.value = 'Working on it...';
    const url = "/detect";
    const request = new XMLHttpRequest();
    request.open('POST', url, true);
    request.onload = function() {
        const img_res = document.getElementById('img_res');
        const new_img = `/get_result?f=${request.responseText}`;
        img_res.src = new_img;
        btn.value = 'Detect those bees!';
    };

    request.onerror = function(err) {
        console.log(err);
    };

    request.send(new FormData(event.target));
    event.preventDefault();
}

const form = document.getElementById('form');
form.addEventListener('submit', imgSubmit);

