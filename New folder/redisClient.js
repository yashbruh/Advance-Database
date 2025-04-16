const redis = require('redis');

// Create Redis client
const client = redis.createClient({
  url: 'redis://localhost:6379' // default Redis port
});

client.on('error', (err) => {
  console.error('Redis error:', err);
});

client.connect();

module.exports = client;
