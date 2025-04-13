const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;

    canvas.addEventListener('mousedown', () => drawing = true);
    canvas.addEventListener('mouseup', () => {
      drawing = false;
      ctx.beginPath();
    });
    canvas.addEventListener('mousemove', draw);

    function draw(e) {
      if (!drawing) return;
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'black';
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.offsetX, e.offsetY);
    }

    function clearCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      document.getElementById('result').innerText = "⬆️ Нарисувай и натисни \"Предскажи\"";
    }

    async function predict() {
      const imageData = ctx.getImageData(0, 0, 280, 280);
      let tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(tf.scalar(255.0))
        .reshape([1, 28, 28, 1]);

      try {
        const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json');
        const prediction = model.predict(tensor);
        const result = prediction.argMax(1);
        const classId = (await result.data())[0];
        document.getElementById('result').innerText = `📢 Предсказание: ${classId}`;
      } catch (err) {
        console.error('❌ Грешка при зареждане или предсказване:', err);
        document.getElementById('result').innerText = '❌ Възникна грешка. Провери конзолата (F12).';
      }
    }

    document.getElementById('predictBtn').addEventListener('click', predict);