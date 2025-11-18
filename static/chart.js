document.addEventListener('DOMContentLoaded', () => {
  if (!window.MODEL_METRICS) return;
  for (const [task, data] of Object.entries(window.MODEL_METRICS)) {
    const ctx = document.getElementById(`chart-${task}`);
    if (!ctx) continue;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          { label: 'F1', data: data.f1, backgroundColor: '#198754' },
          { label: 'Accuracy', data: data.accuracy, backgroundColor: '#0d6efd' },
          { label: 'Recall', data: data.recall, backgroundColor: '#fd7e14' },
        ]
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, max: 1.0 } }
      }
    });
  }
});
