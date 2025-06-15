var Name;

function GetDayFromDate(dateString){
const date = new Date(dateString);
const dayName = date.toLocaleDateString('en-US', { weekday: 'long' });
return dayName
}

function GetCityObject(City) {
    fetch(`https://api.weatherapi.com/v1/forecast.json?key=dbcebcda9c9248f1ac1164956240810&q=${City}&days=4&aqi=no&alerts=yes`)
        .then(response => {
            console.log('Status Code:', response.status);
            if (response.status === 200) {
                return response.json();
            } else {
                throw new Error('Something went wrong');
            }
        })
        .then(data => {
            for (let i = 0; i < data.forecast.forecastday.length; i++) {
                let day = data.forecast.forecastday[i];
                let card = document.getElementById(`day${i + 1}`);
                let iconUrl = `https:${day.day.condition.icon}`;
                card.querySelector('.forecast').innerHTML = `
                ${GetDayFromDate(day.date)}<br>
                <b>${day.day.condition.text}</b><br>
                <img src="${iconUrl}" alt="${day.day.condition.text}" style="width: 64px; height: 64px;"><br>
                Average Temperature: ${day.day.avgtemp_c} °C
            `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function GetName() {
    alert(`Hello dear ${Name}!`)
}

function SetName() {
    Name = prompt("Enter your name:");
    document.getElementById(`username`).innerHTML = Name
}

function GetCity() {
    var CityObject = prompt("Enter the requested city:");
    return CityObject;
}

