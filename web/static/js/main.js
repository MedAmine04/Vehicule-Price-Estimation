// Initialize AOS animations
AOS.init({
    duration: 800,
    once: true
});

// Form handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('estimationForm');
    const results = document.getElementById('results');
    const vehicleTypeInputs = document.querySelectorAll('input[name="vehicle_type"]');
    const brandSelect = document.querySelector('select[name="brand"]');
    const modelSelect = document.querySelector('select[name="model"]');
    const carBrandsGroup = document.getElementById('carBrands');
    const motorcycleBrandsGroup = document.getElementById('motorcycleBrands');
    const carFuelGroup = document.getElementById('carFuel');
    const motorcycleFuelGroup = document.getElementById('motorcycleFuel');
    const submitButton = form.querySelector('button[type="submit"]');
    const seatsInput = form.querySelector('input[name="seats"]');

    // Handle vehicle type change
    vehicleTypeInputs.forEach(input => {
        input.addEventListener('change', function() {
            const isCar = this.value === 'Car';
            
            // Enable/disable brand optgroups
            carBrandsGroup.disabled = !isCar;
            motorcycleBrandsGroup.disabled = isCar;
            
            // Enable/disable fuel type optgroups
            carFuelGroup.disabled = !isCar;
            motorcycleFuelGroup.disabled = isCar;
            
            // Reset selections
            brandSelect.value = '';
            modelSelect.value = '';
            modelSelect.disabled = true;
            
            // Update seats
            if (isCar) {
                seatsInput.value = "5";
            } else {
                seatsInput.value = "2";
            }
        });
    });

    // Handle brand change
    brandSelect.addEventListener('change', function() {
        const selectedBrand = this.value;
        const vehicleType = document.querySelector('input[name="vehicle_type"]:checked').value;
        
        // Clear and disable model select if no brand is selected
        if (!selectedBrand) {
            modelSelect.innerHTML = '<option value="">Select Brand First</option>';
            modelSelect.disabled = true;
            return;
        }
        
        // Get models for selected brand
        const models = vehicleType === 'Car' ? carData[selectedBrand] : motorcycleData[selectedBrand];
        
        // Update model select
        modelSelect.innerHTML = '<option value="">Select Model</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
        
        // Enable model select
        modelSelect.disabled = false;
    });

    // Form validation and submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!form.checkValidity()) {
            e.stopPropagation();
            form.classList.add('was-validated');
            return;
        }

        // Show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="loading-spinner me-2"></span>Estimating...';
        results.style.display = 'none';

        try {
            const formData = new FormData(form);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Format numbers with commas
                const formatNumber = (num) => num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                
                // Update results
                const vehicleType = formData.get('vehicle_type');
                const brand = formData.get('brand');
                const model = formData.get('model');
                
                // Set vehicle details header
                document.getElementById('vehicleDetails').textContent = `${brand} ${model} (${vehicleType})`;
                
                // Set price and confidence interval
                document.getElementById('estimatedPrice').textContent = `${formatNumber(data.prediction)} MAD`;
                document.getElementById('confidenceInterval').textContent = 
                    `${formatNumber(data.confidence_interval[0])} - ${formatNumber(data.confidence_interval[1])} MAD`;
                
                // Update vehicle details
                document.getElementById('resultCity').textContent = formData.get('city');
                document.getElementById('resultYear').textContent = formData.get('year');
                document.getElementById('resultMileage').textContent = formatNumber(formData.get('mileage'));
                document.getElementById('resultTransmission').textContent = formData.get('transmission');
                document.getElementById('resultFuel').textContent = formData.get('fuel');
                document.getElementById('resultAccidents').textContent = formData.get('accidents');
                document.getElementById('resultOwners').textContent = formData.get('owners');
                document.getElementById('resultSeats').textContent = formData.get('seats');
                document.getElementById('resultHorsepower').textContent = formData.get('horsepower');
                
                // Show results with animation
                results.style.display = 'block';
                results.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error connecting to the server. Please try again.');
        } finally {
            // Reset button state
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-calculator me-2"></i>Estimate Price';
        }
    });

    // Set initial values
    seatsInput.value = "5";  // Default for cars
});

// Helper function to format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('fr-MA', {
        style: 'currency',
        currency: 'MAD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
}); 