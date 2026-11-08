import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Initialize Gemini API lazily
  const getGeminiClient = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is not configured in environment secrets.');
    }
    return new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  };

  // API Route: AI Sensory Analysis using Gemini
  app.post('/api/gemini/analyze-taste', async (req, res) => {
    try {
      const { sampleType, features } = req.body;
      const ai = getGeminiClient();

      const prompt = `You are an expert Flavor Chemist and Sommelier specializing in Artificial Taste Perception and Electronic Tongue (E-Tongue) modeling.
Analyze the following ${sampleType} physicochemical & sensory parameters:
${JSON.stringify(features, null, 2)}

Provide a concise, professional analysis in Markdown:
1. **Perceived Taste Notes**: (Sourness, Sweetness, Bitterness, Umami, Astringency balance)
2. **Chem sensory Drivers**: Key chemical compounds driving this taste profile.
3. **Quality & Formulation Advice**: Actionable recommendation to improve taste quality.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.6-flash',
        contents: prompt,
      });

      res.json({ analysis: response.text });
    } catch (error: any) {
      console.error('Gemini API Error:', error);
      res.status(500).json({ error: error.message || 'Failed to generate AI sensory analysis' });
    }
  });

  // API Route: List files in Artificial_Taste_Project
  app.get('/api/project/files', (req, res) => {
    try {
      const baseDir = path.join(process.cwd(), 'Artificial_Taste_Project');
      
      const getTree = (dir: string, relPath = ''): any[] => {
        if (!fs.existsSync(dir)) return [];
        const items = fs.readdirSync(dir, { withFileTypes: true });
        return items.map((item) => {
          const itemRel = relPath ? `${relPath}/${item.name}` : item.name;
          const itemFull = path.join(dir, item.name);
          if (item.isDirectory()) {
            return {
              name: item.name,
              path: itemRel,
              type: 'directory',
              children: getTree(itemFull, itemRel),
            };
          } else {
            return {
              name: item.name,
              path: itemRel,
              type: 'file',
              size: fs.statSync(itemFull).size,
            };
          }
        });
      };

      const tree = getTree(baseDir);
      res.json({ tree });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // API Route: Get specific file content
  app.get('/api/project/file-content', (req, res) => {
    try {
      const filePath = req.query.path as string;
      if (!filePath) return res.status(400).json({ error: 'Missing path query param' });

      const fullPath = path.join(process.cwd(), 'Artificial_Taste_Project', filePath);
      if (!fs.existsSync(fullPath)) {
        return res.status(404).json({ error: 'File not found' });
      }

      const content = fs.readFileSync(fullPath, 'utf-8');
      res.json({ content, name: path.basename(fullPath) });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Vite development or production middleware
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
