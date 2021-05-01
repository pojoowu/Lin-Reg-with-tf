const x_train_vals = [],
    y_train_vals = [];
const learningRate = 0.5,
    optimizer = tf.train.sgd(learningRate);

const slope = tf.variable(tf.scalar(Math.random(1)));
const y0 = tf.variable(tf.scalar(Math.random(1)));
//output tensors while input is array
function predict(x_vals) {
    const tfxs = tf.tensor1d(x_vals);
    return tfxs.mul(slope).add(y0);
}

function loss(guesses, lables) {
    return guesses.sub(lables).square().mean();
}

function setup() {
    createCanvas(600, 400);
}

async function draw() {
    frameRate(10);
    background(51);
    for (let i = 0; i < x_train_vals.length; i++) {
        let x = map(x_train_vals[i], 0, 1, 0, width);
        let y = map(y_train_vals[i], 0, 1, height, 0);
        strokeWeight(10);
        stroke(255);
        point(x, y);
    }
    if (x_train_vals.length > 0) {
        tf.tidy(() => {
            let tfys = tf.tensor1d(y_train_vals);
            optimizer.minimize(() => loss(predict(x_train_vals), tfys));
        });
    }

    let x_predict_vals = [0, 1];
    let tfy_predict_vals = tf.tidy(() => predict(x_predict_vals));
    let y_predict_vals = await tfy_predict_vals.data();
    tfy_predict_vals.dispose();

    for(let i = 0; i<2;i++){
        x_predict_vals[i] = map(x_predict_vals[i], 0, 1, 0, width);
        y_predict_vals[i] = map(y_predict_vals[i], 0, 1, height, 0);
    }
    strokeWeight(3);
    stroke(255);
    line(x_predict_vals[0], y_predict_vals[0], x_predict_vals[1], y_predict_vals[1]);
    noStroke();
    fill(255);
    rectMode(CORNER);
    textSize(24);
    text(`${await slope.data()}`, 0, 24);
}

function mousePressed() {
    if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
        let x = map(mouseX, 0, width, 0, 1);
        let y = map(mouseY, 0, height, 1, 0);
        x_train_vals.push(x);
        y_train_vals.push(y);
    }
}