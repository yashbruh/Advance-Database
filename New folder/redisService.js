// redisService.js
const redis = require('redis');

// Connect to Redis server (Assuming Redis is running locally)
const client = redis.createClient({
  url: 'redis://localhost:6379'
});

// Error handling
client.on('error', (err) => {
  console.error('Redis Client Error:', err);
});

// Connecting to Redis (you can add a connection handler here)
(async () => {
  await client.connect();
})();

// Function to store car data in Redis
async function storeCarData(carId, data) {
  try {
    await client.set(`car:${carId}`, JSON.stringify(data));
    console.log(`Car data with ID ${carId} stored successfully.`);
  } catch (error) {
    console.error('Error storing car data:', error);
  }
}

// Function to retrieve car data from Redis
async function getCarData(carId) {
  try {
    const data = await client.get(`car:${carId}`);
    return JSON.parse(data);
  } catch (error) {
    console.error('Error retrieving car data:', error);
    return null;
  }
}

// Export the functions to be used in other files
module.exports = {
  storeCarData,
  getCarData
};
