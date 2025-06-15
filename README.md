# Streamlit App Deployment on AWS

A complete guide to deploy your Streamlit application on AWS EC2 with custom domain and HTTPS.

## 🚀 Live Demo

Check out the live application: [https://akshitmlstudio.click](https://akshitmlstudio.click)

## 📋 Prerequisites

Before starting, make sure you have:
- AWS Account (with free tier eligible)
- Docker Hub account
- Basic familiarity with command line
- Git installed on your local machine

## 🛠️ Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Akshit-77/Personality-predictor-app.git
cd Personality-predictor-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```
Visit `http://localhost:8501` to view your app.

## 🐳 Docker Setup

### 1. Build Docker Image
```bash
# Build the Docker image
docker build -t your-dockerhub-username/your-app-name:latest .

# Test locally
docker run -p 8501:8501 your-dockerhub-username/your-app-name:latest
```

### 2. Push to Docker Hub
```bash
# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/your-app-name:latest
```

## ☁️ AWS EC2 Deployment

### Step 1: Create EC2 Instance

1. **Login to AWS Console** → EC2 → Launch Instance
2. **Configure Instance:**
   - **Name**: `streamlit-app-server`
   - **OS**: Ubuntu Server 22.04 LTS
   - **Instance type**: `t2.micro` (free tier)
   - **Key pair**: Create new or use existing
   - **Network Settings**:
     - Create security group with these rules:
     - SSH (port 22) from 0.0.0.0/0
     - HTTP (port 80) from 0.0.0.0/0
     - HTTPS (port 443) from 0.0.0.0/0
     - Custom TCP (port 8501) from 0.0.0.0/0
3. **Launch Instance**

### Step 2: Allocate Elastic IP

1. **EC2 Console** → Elastic IPs → Allocate Elastic IP
2. **Associate** with your EC2 instance
3. **Note down the Elastic IP** (e.g., 13.233.xxx.xxx)

### Step 3: Connect to EC2

**Option A: Using PuTTY (Windows)**
1. Convert `.pem` to `.ppk` using PuTTYgen
2. Connect using PuTTY with your `.ppk` file

**Option B: Using SSH**
```bash
ssh -i your-key.pem ubuntu@your-elastic-ip
```

**Option C: AWS Console**
- EC2 → Instances → Connect → EC2 Instance Connect

### Step 4: Install Docker on EC2

```bash
# Update system
sudo apt update -y

# Install Docker
sudo apt install docker.io -y

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add ubuntu user to docker group
sudo usermod -a -G docker ubuntu

# Apply group changes
newgrp docker

# Verify installation
docker --version
```

### Step 5: Deploy Your App

```bash
# Pull your Docker image
docker pull your-dockerhub-username/your-app-name:latest

# Run the container
docker run -d \
  --name streamlit-app \
  -p 8501:8501 \
  --restart unless-stopped \
  your-dockerhub-username/your-app-name:latest

# Verify it's running
docker ps
```

**Test**: Visit `http://your-elastic-ip:8501` - your app should be live!

## 🌐 Custom Domain Setup (Route 53)

### Step 1: Register Domain

1. **AWS Console** → Route 53 → Register domain
2. Choose your domain (e.g., `yourapp.com`)
3. Complete registration process

### Step 2: Create DNS Records

1. **Route 53** → Hosted zones → Select your domain
2. **Create A Record for root domain:**
   - Record name: [leave blank]
   - Type: A
   - Value: Your Elastic IP
   - TTL: 300

3. **Create A Record for www:**
   - Record name: www
   - Type: A
   - Value: Your Elastic IP
   - TTL: 300

### Step 3: Set Up Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install nginx -y

# Start and enable Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Create site configuration
sudo nano /etc/nginx/sites-available/yourapp
```

Add this configuration (replace `yourapp.com` with your domain):

```nginx
server {
    listen 80;
    server_name yourapp.com www.yourapp.com;
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
        proxy_buffering off;
    }
    
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable the site:
```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/yourapp /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

**Test**: Visit `http://yourapp.com` - should work without port number!

## 🔒 Add HTTPS (SSL Certificate)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d yourapp.com -d www.yourapp.com

# Follow prompts and choose option 2 (redirect HTTP to HTTPS)

# Set up auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Test auto-renewal
sudo certbot renew --dry-run
```

**Final Result**: Your app is now live at `https://yourapp.com` with automatic HTTPS redirect!

## 🔄 Updating Your App

When you make code changes:

### 1. Update and Push
```bash
# Make your changes locally
# Test with: streamlit run app.py

# Build new Docker image
docker build -t your-dockerhub-username/your-app-name:latest .

# Push to Docker Hub
docker push your-dockerhub-username/your-app-name:latest
```

### 2. Update on EC2
```bash
# Connect to your EC2 instance
# Pull latest image
docker pull your-dockerhub-username/your-app-name:latest

# Stop and remove old container
docker stop streamlit-app
docker rm streamlit-app

# Run updated container
docker run -d \
  --name streamlit-app \
  -p 8501:8501 \
  --restart unless-stopped \
  your-dockerhub-username/your-app-name:latest
```

### Create Update Script (Optional)
Create `update-app.sh` on your EC2:
```bash
#!/bin/bash
echo "Updating app..."
docker pull your-dockerhub-username/your-app-name:latest
docker stop streamlit-app
docker rm streamlit-app
docker run -d --name streamlit-app -p 8501:8501 --restart unless-stopped your-dockerhub-username/your-app-name:latest
echo "App updated successfully!"
```

Make it executable and use:
```bash
chmod +x update-app.sh
./update-app.sh
```

### Domain not working
```bash
# Check DNS resolution
nslookup yourapp.com

# Check Nginx status
sudo systemctl status nginx

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### SSL issues
```bash
# Check certificate status
sudo certbot certificates

# Renew certificate manually
sudo certbot renew

# Check renewal timer
sudo systemctl status certbot.timer
```

## 📁 Project Structure

```
your-repo/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Acknowledgments

- Thanks to the Streamlit community
- AWS Free Tier for making this accessible
- Let's Encrypt for free SSL certificates

---

**🎉 Congratulations!** You now have a production-ready Streamlit app deployed on AWS with a custom domain and HTTPS!

**Live Demo**: [https://akshitmlstudio.click](https://akshitmlstudio.click)
