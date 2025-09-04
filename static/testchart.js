// Initialize the Pie Chart
const ctx = document.getElementById('pieChart').getContext('2d');
const pieChart = new Chart(ctx, {
    type: 'pie',  // Pie chart type
    data: {
        labels: ['Real', 'Fake'],  // Labels for segments
        datasets: [{
            data: [70, 30],  // Data values (e.g., 70% Real, 30% Fake)
            backgroundColor: ['#4CAF50', '#F44336'],  // Color for each segment
            borderColor: ['#ffffff', '#ffffff'],  // Border color for each segment
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,  // Make the chart responsive
        plugins: {
            legend: {
                position: 'top',  // Position the legend on top
            },
        },
    }
});