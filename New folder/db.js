const { Pool } = require('pg');
const redis = require('redis');
const mongoose = require('mongoose');

// PostgreSQL connection (if you're still using it)
const pool = new Pool({
  user: 'your_pg_username',
  host: 'localhost',
  database: 'your_pg_database',
  password: 'your_pg_password',
  port: 5432,
});

// Redis setup
const redisClient = redis.createClient({
  socket: {
    host: 'localhost',
    port: 6379
  }
});

redisClient.connect()
  .then(() => console.log('Connected to Redis'))
  .catch((err) => console.error('Redis connection error:', err));

// MongoDB setup
mongoose.connect('mongodb://localhost:27017/trafficData', {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.log('MongoDB connection error:', err));

module.exports = { pool, redisClient, mongoose };
