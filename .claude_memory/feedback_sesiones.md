---
name: Preferencias de trabajo en sesiones
description: Cómo el usuario quiere que se maneje el contexto entre sesiones
type: feedback
originSessionId: eec95db5-34d2-4ecf-9efc-40e0b839e51a
---
Al iniciar una nueva sesión, NO pedir que el usuario describa el proyecto desde cero. En cambio:
1. Leer la memoria del proyecto (`project_translow_distill.md`) para tener contexto completo
2. Revisar el git log reciente y archivos modificados para entender el estado actual
3. Preguntar solo: **¿qué archivo o tarea específica se estaba trabajando y qué cambio faltaba terminar?**

**Why:** El usuario quiere continuar el trabajo sin repetir instrucciones en cada sesión. Se molesta cuando tiene que reexplicar el proyecto desde el inicio.

**How to apply:** En cada sesión nueva, usar la memoria para reconstruir el contexto y hacer preguntas precisas y cortas para retomar el trabajo pendiente.
