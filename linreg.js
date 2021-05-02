const x_train_vals = [],
    y_train_vals = [];
const learningRate = 0.5,
    optimizer = tf.train.adam(learningRate);
let slope, y0, degree = 1;
let parameters = [];
let input, button;
const scale = 0.05;


let y_predict_vals = [0, 1];
const x_predict_vals = [];
for (let x = 0; x <= 1.01; x += scale) {
    x_predict_vals.push(x);
}
let x_real_vals = [],
    y_real_vals = [];

function setup() {
    createCanvas(400, 400);
    input = createInput('');
    button = createButton('Reset');
    parameters.push(tf.variable(tf.scalar(0)));
    parameters.push(tf.variable(tf.scalar(1)));
    tf.tidy(() => {
        setInterval(training, 30);
        setInterval(getYs, 30);
    });
}

async function draw() {
    background(51);
    button.mousePressed(reset);
    for (let i = 0; i < x_train_vals.length; i++) {
        let x = map(x_train_vals[i], 0, 1, 0, width);
        let y = map(y_train_vals[i], 0, 1, height, 0);
        strokeWeight(10);
        stroke(255);
        point(x, y);
    }
    strokeWeight(3);
    stroke(255);
    beginShape();
    noFill();
    for (let i = 0; i < x_predict_vals.length; i++) {
        let realX = map(x_predict_vals[i], 0, 1, 0, width);
        let realY = map(y_predict_vals[i], 0, 1, height, 0);
        vertex(realX, realY);
    }
    endShape();
}

function mousePressed() {
    if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
        let x = map(mouseX, 0, width, 0, 1);
        let y = map(mouseY, 0, height, 1, 0);
        x_train_vals.push(x);
        y_train_vals.push(y);
    }
}

function mouseDragged() {
    if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
        let x = map(mouseX, 0, width, 0, 1);
        let y = map(mouseY, 0, height, 1, 0);
        x_train_vals.push(x);
        y_train_vals.push(y);
    }
}

//output tensors while input is array
function predict(x_vals) {
    return tf.tidy(() => {
        const tfxs = tf.tensor1d(x_vals);

        let tfxs1 = tf.tensor1d(x_vals.map(el => 0));
        for (let i = degree; i >= 0; i--) {
            tfxs1 = tfxs1.mul(tfxs).add(parameters[i]);
        }
        //console.log(console.log(tf.memory().numTensors));
        return tfxs1;
    });
}

function loss(guesses, lables) {
    return guesses.sub(lables).square().mean();
}

function training() {
    if (x_train_vals.length > 0) {
        tf.tidy(() => {
            let tfys = tf.tensor1d(y_train_vals);
            optimizer.minimize(() => loss(predict(x_train_vals), tfys));
            //console.log(x_train_vals);
        });
        //console.log(tf.memory().numTensors);
    }
}

async function getYs() {
    let tfy_predict_vals = predict(x_predict_vals);
    y_predict_vals = await tfy_predict_vals.data();
    //console.log(y_predict_vals);
    tfy_predict_vals.dispose();
}

function reset() {
    if (input.value() > 0) {
        degree = input.value();
        tf.dispose(parameters);
        tf.dispose(parameters);
        parameters = [];
        for (let i = 0; i <= degree; i++) {
            if (i !== 1) {
                parameters[i] = tf.variable(tf.scalar(0));
            } else {
                parameters[i] = tf.variable(tf.scalar(1));
            }
        }
        x_train_vals.length = 0;
        y_train_vals.length = 0;
    }
}