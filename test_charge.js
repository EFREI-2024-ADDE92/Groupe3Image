import http from 'k6/http';
import { sleep } from 'k6';

export let options = {
  vus: 100,
  duration: '50s',
};

export default function () {
  const url = 'https://group3-image-container.blackocean-8e9ea989.francecentral.azurecontainerapps.io/predict';

  const payload = 
    { file : 'forest.png' };

  const headers = {
    'Content-Type': 'image/png'
  };

  const response = http.post(url, JSON.stringify(payload), { headers: headers });

  sleep(1);
}