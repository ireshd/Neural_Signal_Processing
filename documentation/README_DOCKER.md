# Docker Setup for Neural Signal DSP Pipeline

This guide explains how to run the Neural Signal DSP project using Docker.

## Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Build and run the container:
```bash
docker-compose up --build
```

2. Run with a specific script:
```bash
docker-compose run neural-dsp python src/signal_gen.py
```

3. Stop the container:
```bash
docker-compose down
```

### Option 2: Using Docker CLI

1. Build the Docker image:
```bash
docker build -t neural-signal-dsp .
```

2. Run the signal generator:
```bash
docker run --rm -v "%cd%/data:/app/data" neural-signal-dsp
```

On Linux/Mac, use:
```bash
docker run --rm -v "$(pwd)/data:/app/data" neural-signal-dsp
```

3. Run a specific script:
```bash
docker run --rm -v "%cd%/data:/app/data" neural-signal-dsp python src/adc_sim.py
```

## Viewing Output Files

Generated plots and CSV data files are saved to the `./data/outputs` directory, which is mounted as a volume. You can view them directly on your host machine.

**Output files include:**
- PNG plots (signal visualizations)
- CSV data files (signal data, spike times, waveforms, statistics)

See `CSV_OUTPUT_FORMAT.md` for details on CSV file formats.

## Interactive Development

To run the container interactively for development:

```bash
docker run -it --rm -v "%cd%:/app" neural-signal-dsp /bin/bash
```

On Linux/Mac:
```bash
docker run -it --rm -v "$(pwd):/app" neural-signal-dsp /bin/bash
```

This gives you a shell inside the container where you can run Python scripts and test code.

## Running Examples

### Signal Generator with CSV Export

```bash
docker-compose run neural-dsp python src/signal_gen.py
```

This will generate:
- PNG plots in `data/outputs/neural_signal_demo.png`
- CSV data files in `data/outputs/*.csv`

### CSV Export Examples

```bash
docker-compose run neural-dsp python examples/csv_export_example.py
```

### Complete Pipeline

Once all modules are implemented:

```bash
docker-compose run neural-dsp python src/realtime_loop.py
```

## Troubleshooting

### Permission Issues (Linux)

If you encounter permission issues with mounted volumes:

```bash
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)/data:/app/data" neural-signal-dsp
```

### Display/GUI Issues

The container uses a non-interactive matplotlib backend (Agg) by default, which saves plots to files instead of displaying them. This is ideal for Docker environments without a display.

If you need interactive plotting with X11 forwarding (Linux/Mac):

```bash
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)/data:/app/data" neural-signal-dsp
```

On Windows, you'll need an X server like VcXsrv or Xming.

## Building for Production

To build an optimized production image:

```bash
docker build -t neural-signal-dsp:prod --target production .
```

## Environment Variables

Available environment variables:

- `MPLBACKEND`: Matplotlib backend (default: `Agg`)
- `PYTHONUNBUFFERED`: Enable unbuffered Python output (default: `1`)

Example with custom settings:

```bash
docker run --rm -e MPLBACKEND=TkAgg -v "%cd%/data:/app/data" neural-signal-dsp
```

## Clean Up

Remove all containers and images:

```bash
docker-compose down --rmi all
docker system prune -a
```

