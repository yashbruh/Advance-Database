
# Redis Project Setup

## Steps to Run Redis and Test the Database

### Files Included:
1. `server.js`: Main server file that connects to Redis and handles requests.
2. `redisClient.js`: Configuration file for connecting to Redis.
3. `package.json`: Project dependencies, including Redis client.
4. `redisService.js`: Functions for interacting with Redis.
5. `db.js`: Handles database-specific logic (if applicable).
6. Other necessary project files.

### Steps to Test:

1. **Install Redis**:
   - Download and install Redis from [redis.io](https://redis.io/download).
   - After installation, start the Redis server by running:
     ```bash
     redis-server
     ```

2. **Install Node.js**:
   - Ensure Node.js is installed. If not, download it from [nodejs.org](https://nodejs.org/).

3. **Install Project Dependencies**:
   - Navigate to the project directory and run the following command to install the dependencies:
     ```bash
     npm install
     ```

4. **Start Redis**:
   - Ensure Redis is running with the command:
     ```bash
     redis-server
     ```

5. **Start the Node.js Server**:
   - After setting up Redis, start the server with:
     ```bash
     node server/server.js
     ```

6. **Testing the API**:
   - Use Postman or cURL to test the following routes:
     - **POST** `/store-detection`: Store vehicle detection data.
     - **GET** `/get-detection/:vehicleId`: Retrieve detection data.

Example for storing detection data:
```bash
POST http://localhost:5000/store-detection
Content-Type: application/json

{
  "vehicleId": "car001",
  "snapshot": "imageData",
  "timestamp": "2025-04-16T12:00:00Z",
  "location": "Location details"
}
```

Example for retrieving detection data:
```bash
GET http://localhost:5000/get-detection/car001
```

## Troubleshooting:
- Ensure Redis is running before starting the Node.js server.
- Check for any errors in the server or Redis logs.
