# cli.py - Command Line Interface for Dual Brain AI System

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import the master orchestrator
try:
    from unified_orchestration import UnifiedOrchestrationSystem, PipelineManager, AutonomousOrchestrator, SystemMode
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Unified orchestration not available: {e}")
    ORCHESTRATOR_AVAILABLE = False

class DualBrainCLI:
    """Command Line Interface for the Dual Brain AI System"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.orchestrator: Optional[UnifiedOrchestrationSystem] = None
        self.verbose = False
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Dual Brain AI System - Autonomous AI with Logic/Symbolic Intelligence",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s start --mode autonomous           # Start in autonomous mode
  %(prog)s start --mode interactive         # Start interactive chat mode
  %(prog)s status                           # Show system status
  %(prog)s migrate                          # Run migration cycle
  %(prog)s integrate                        # Run integration cycle
  %(prog)s health                           # Check system health
  %(prog)s backup                           # Create system backup
  %(prog)s stop                             # Stop all subsystems
            """
        )
        
        # Global options
        parser.add_argument('--data-dir', '-d', default='data', 
                          help='Data directory path (default: data)')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose output')
        parser.add_argument('--config', '-c', default='config.json',
                          help='Configuration file path (default: config.json)')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Start command
        start_parser = subparsers.add_parser('start', help='Start the AI system')
        start_parser.add_argument('--mode', '-m', choices=['autonomous', 'interactive', 'learning', 'integration', 'maintenance', 'legacy'],
                                default='autonomous', help='System mode to start in (default: autonomous)')
        start_parser.add_argument('--input', '-i', help='Process this input immediately after start')
        start_parser.add_argument('--urls', type=int, default=10, help='Number of URLs to process in autonomous mode')
        
        # Status command
        subparsers.add_parser('status', help='Show system status')
        
        # Stop command
        subparsers.add_parser('stop', help='Stop all running subsystems')
        
        # Migration commands
        subparsers.add_parser('migrate', help='Run unified migration cycle')
        
        # Integration commands
        subparsers.add_parser('integrate', help='Run integration cycle')
        
        # Data analysis
        subparsers.add_parser('analyze', help='Run data analysis')
        
        # Health check
        subparsers.add_parser('health', help='Check system health')
        
        # Backup
        subparsers.add_parser('backup', help='Create system backup')
        
        # Cleanup
        subparsers.add_parser('cleanup', help='Clean up old files')
        
        # Interactive mode
        interact_parser = subparsers.add_parser('interact', help='Start interactive session')
        interact_parser.add_argument('--continuous', action='store_true',
                                   help='Keep session running until manually stopped')
        
        # Learning mode
        learn_parser = subparsers.add_parser('learn', help='Start learning mode')
        learn_parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=1,
                                help='Learning phase to focus on (default: 1)')
        learn_parser.add_argument('--urls', type=int, default=10,
                                help='Number of URLs to process (default: 10)')
        
        # Chat mode
        chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
        chat_parser.add_argument('message', nargs='?', help='Send a message directly')
        
        # Config commands
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')
        config_subparsers.add_parser('show', help='Show current configuration')
        config_subparsers.add_parser('reset', help='Reset configuration to defaults')
        
        # Memory commands
        memory_parser = subparsers.add_parser('memory', help='Memory management')
        memory_subparsers = memory_parser.add_subparsers(dest='memory_action', help='Memory actions')
        memory_subparsers.add_parser('stats', help='Show memory statistics')
        memory_subparsers.add_parser('export', help='Export memory data')
        memory_subparsers.add_parser('import', help='Import memory data')
        
        return parser
    
    def print_status(self, message: str, level: str = "info"):
        """Print status message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "error":
            icon = "❌"
        elif level == "warning":
            icon = "⚠️ "
        elif level == "success":
            icon = "✅"
        elif level == "info":
            icon = "ℹ️ "
        else:
            icon = "  "
        
        print(f"[{timestamp}] {icon} {message}")
    
    def ensure_orchestrator(self) -> bool:
        """Ensure orchestrator is initialized"""
        if not ORCHESTRATOR_AVAILABLE:
            self.print_status("Master orchestrator not available", "error")
            return False
            
        if self.orchestrator is None:
            try:
                self.print_status(f"Initializing orchestrator with data directory: {self.data_dir}")
                self.orchestrator = UnifiedOrchestrationSystem(data_dir=self.data_dir)
                self.print_status("Orchestrator initialized successfully", "success")
                return True
            except Exception as e:
                self.print_status(f"Failed to initialize orchestrator: {e}", "error")
                return False
        return True
    
    def cmd_start(self, args) -> int:
        """Start the AI system"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            mode = SystemMode(args.mode)
            self.print_status(f"Starting system in {mode.value} mode...")
            
            # Prepare kwargs based on mode
            kwargs = {}
            if mode == SystemMode.AUTONOMOUS:
                kwargs['target_urls'] = args.urls
            
            result = self.orchestrator.start_system(mode, **kwargs)
            
            if result.get('status') == 'error':
                self.print_status(f"Failed to start system: {result.get('error')}", "error")
                return 1
            
            self.print_status(f"System started successfully in {mode.value} mode", "success")
            
            if self.verbose:
                subsystems = result.get('subsystems_started', [])
                if subsystems:
                    self.print_status(f"Active subsystems: {', '.join(subsystems)}")
            
            # Process immediate input if provided
            if args.input:
                self.print_status(f"Processing input: {args.input}")
                input_result = self.orchestrator.process_input(args.input)
                print(json.dumps(input_result, indent=2))
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error starting system: {e}", "error")
            return 1
    
    def cmd_status(self, args) -> int:
        """Show system status"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            status = self.orchestrator.get_system_status()
            
            print("\n" + "="*60)
            print("🤖 DUAL BRAIN AI SYSTEM STATUS")
            print("="*60)
            
            print(f"\n🔄 Current Mode: {status.mode.value}")
            print(f"📊 Status: {status.status.value}")
            print(f"⏱️  Uptime: {status.uptime:.1f} seconds")
            
            if status.active_subsystems:
                print(f"\n✅ Active Subsystems ({len(status.active_subsystems)}):")
                for subsystem in status.active_subsystems:
                    print(f"   • {subsystem}")
            else:
                print("\n💤 No active subsystems")
            
            print(f"\n💾 Memory Usage:")
            mem = status.memory_usage
            print(f"   Data files: {mem.get('data_files', 0)}")
            print(f"   Total size: {mem.get('total_size_mb', 0):.1f} MB")
            
            print(f"\n🔒 Security Status:")
            sec = status.security_status
            print(f"   AlphaWall: {'✅' if sec.get('alphawall_available') else '❌'}")
            print(f"   Quarantine: {'✅' if sec.get('quarantine_active') else '❌'}")
            
            if status.warnings:
                print(f"\n⚠️  Warnings ({len(status.warnings)}):")
                for warning in status.warnings:
                    print(f"   • {warning}")
            
            if status.errors:
                print(f"\n❌ Errors ({len(status.errors)}):")
                for error in status.errors:
                    print(f"   • {error}")
            
            print(f"\n📈 Performance:")
            perf = status.performance_metrics
            print(f"   Session ID: {perf.get('session_id', 'unknown')}")
            print(f"   Current mode: {perf.get('current_mode', 'none')}")
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error getting status: {e}", "error")
            return 1
    
    def cmd_stop(self, args) -> int:
        """Stop all running subsystems"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Stopping all subsystems...")
            result = self.orchestrator.stop_system()
            
            stopped = result.get('stopped_subsystems', [])
            if stopped:
                self.print_status(f"Stopped {len(stopped)} subsystems: {', '.join(stopped)}", "success")
            else:
                self.print_status("No subsystems were running")
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error stopping system: {e}", "error")
            return 1
    
    def cmd_migrate(self, args) -> int:
        """Run migration cycle"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Starting migration cycle...")
            result = self.orchestrator.execute_command('migration_cycle')
            
            if result.get('status') == 'completed':
                self.print_status("Migration cycle completed successfully", "success")
                if self.verbose:
                    print(f"   Session ID: {result.get('session_id')}")
                    print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
            else:
                self.print_status(f"Migration cycle failed: {result.get('error')}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error running migration: {e}", "error")
            return 1
    
    def cmd_integrate(self, args) -> int:
        """Run integration cycle"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Starting integration cycle...")
            result = self.orchestrator.execute_command('integration_cycle')
            
            if result.get('status') == 'completed':
                self.print_status("Integration cycle completed successfully", "success")
                if self.verbose:
                    print(f"   Cycle ID: {result.get('cycle_id')}")
                    print(f"   Integration score: {result.get('integration_score', 0):.2f}")
            else:
                self.print_status(f"Integration cycle failed: {result.get('error')}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error running integration: {e}", "error")
            return 1
    
    def cmd_analyze(self, args) -> int:
        """Run data analysis"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Running data analysis...")
            result = self.orchestrator.execute_command('data_analysis')
            
            if result.get('status') == 'completed':
                self.print_status("Data analysis completed successfully", "success")
                print(f"   Total files analyzed: {result.get('total_files', 0)}")
                print(f"   Total data size: {result.get('total_size_mb', 0):.1f} MB")
            else:
                self.print_status(f"Data analysis failed: {result.get('error')}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error running analysis: {e}", "error")
            return 1
    
    def cmd_health(self, args) -> int:
        """Check system health"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Checking system health...")
            result = self.orchestrator.execute_command('system_health')
            
            print(f"\n🏥 System Health: {result.get('overall_status', 'unknown').upper()}")
            print(f"   Timestamp: {result.get('timestamp', 'unknown')}")
            print(f"   Data files: {result.get('data_files', 0)}")
            
            subsystems = result.get('subsystems', {})
            if subsystems:
                print(f"\n🔧 Subsystem Status:")
                for name, status in subsystems.items():
                    available = "✅" if status.get('available') else "❌"
                    active = "🟢" if status.get('active') else "⭕"
                    print(f"   {available} {active} {name}")
            
            warnings = result.get('warnings', [])
            if warnings:
                print(f"\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
            
            errors = result.get('errors', [])
            if errors:
                print(f"\n❌ Errors:")
                for error in errors:
                    print(f"   • {error}")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error checking health: {e}", "error")
            return 1
    
    def cmd_backup(self, args) -> int:
        """Create system backup"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Creating system backup...")
            result = self.orchestrator.execute_command('backup')
            
            if result.get('status') == 'completed':
                self.print_status("Backup created successfully", "success")
                print(f"   Location: {result.get('backup_dir')}")
                print(f"   Files backed up: {result.get('files_backed_up', 0)}")
            else:
                self.print_status(f"Backup failed: {result.get('error')}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error creating backup: {e}", "error")
            return 1
    
    def cmd_cleanup(self, args) -> int:
        """Clean up old files"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status("Running system cleanup...")
            result = self.orchestrator.execute_command('cleanup')
            
            if result.get('status') == 'completed':
                cleaned = result.get('cleaned_items', [])
                if cleaned:
                    self.print_status(f"Cleanup completed - removed {len(cleaned)} items", "success")
                    if self.verbose:
                        for item in cleaned:
                            print(f"   • {item}")
                else:
                    self.print_status("No cleanup needed")
            else:
                self.print_status(f"Cleanup failed: {result.get('error')}", "error")
                return 1
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error during cleanup: {e}", "error")
            return 1
    
    def cmd_interact(self, args) -> int:
        """Start interactive session"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            # Start in interactive mode
            mode_result = self.orchestrator.start_system(SystemMode.INTERACTIVE)
            if mode_result.get('status') == 'error':
                self.print_status(f"Failed to start interactive mode: {mode_result.get('error')}", "error")
                return 1
            
            self.print_status("Interactive session started", "success")
            print("\nType 'exit' to end the session")
            print("-" * 50)
            
            while True:
                try:
                    user_input = input("\n🗣️  You: ").strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    
                    if not user_input:
                        continue
                    
                    # Process input
                    result = self.orchestrator.process_input(user_input)
                    
                    if result.get('error'):
                        print(f"❌ Error: {result['error']}")
                    else:
                        response = result.get('response', 'No response generated')
                        print(f"\n🤖 AI: {response}")
                        
                except KeyboardInterrupt:
                    print("\n\nSession interrupted. Type 'exit' to quit properly.")
                    break
                except EOFError:
                    break
            
            # Stop the session
            self.orchestrator.stop_system()
            self.print_status("Interactive session ended")
            return 0
            
        except Exception as e:
            self.print_status(f"Error in interactive session: {e}", "error")
            return 1
    
    def cmd_learn(self, args) -> int:
        """Start learning mode"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            self.print_status(f"Starting learning mode (Phase {args.phase})...")
            
            # Start in learning mode
            mode_result = self.orchestrator.start_system(SystemMode.LEARNING, 
                                                        phase=args.phase, 
                                                        urls=args.urls)
            
            if mode_result.get('status') == 'error':
                self.print_status(f"Failed to start learning mode: {mode_result.get('error')}", "error")
                return 1
            
            self.print_status("Learning mode started successfully", "success")
            
            # Learning mode typically runs autonomously
            self.print_status("Learning is running autonomously...")
            self.print_status("Use 'status' command to monitor progress")
            
            return 0
            
        except Exception as e:
            self.print_status(f"Error starting learning mode: {e}", "error")
            return 1
    
    def cmd_chat(self, args) -> int:
        """Start chat mode or send a single message"""
        if not self.ensure_orchestrator():
            return 1
            
        try:
            # Start in interactive mode
            mode_result = self.orchestrator.start_system(SystemMode.INTERACTIVE)
            if mode_result.get('status') == 'error':
                self.print_status(f"Failed to start chat mode: {mode_result.get('error')}", "error")
                return 1
            
            if args.message:
                # Send single message
                result = self.orchestrator.process_input(args.message)
                
                if result.get('error'):
                    print(f"❌ Error: {result['error']}")
                    return 1
                else:
                    response = result.get('response', 'No response generated')
                    print(f"🤖 AI: {response}")
            else:
                # Interactive chat
                self.print_status("Chat mode started - type your messages", "success")
                print("Type 'exit' to end the chat")
                print("-" * 50)
                
                while True:
                    try:
                        user_input = input("\n💬 You: ").strip()
                        
                        if user_input.lower() in ['exit', 'quit']:
                            break
                        
                        if not user_input:
                            continue
                        
                        result = self.orchestrator.process_input(user_input)
                        
                        if result.get('error'):
                            print(f"❌ Error: {result['error']}")
                        else:
                            response = result.get('response', 'No response generated')
                            print(f"\n🤖 AI: {response}")
                            
                    except KeyboardInterrupt:
                        print("\n\nChat interrupted. Type 'exit' to quit properly.")
                        break
                    except EOFError:
                        break
            
            # Stop the session
            self.orchestrator.stop_system()
            return 0
            
        except Exception as e:
            self.print_status(f"Error in chat mode: {e}", "error")
            return 1
    
    def cmd_config(self, args) -> int:
        """Configuration management"""
        if args.config_action == 'show':
            if self.ensure_orchestrator():
                config = self.orchestrator.config
                print("\n📋 Current Configuration:")
                print(json.dumps(config, indent=2))
                return 0
            return 1
        elif args.config_action == 'reset':
            # Reset configuration logic would go here
            self.print_status("Configuration reset (not implemented)", "warning")
            return 0
        else:
            self.print_status("No config action specified", "error")
            return 1
    
    def cmd_memory(self, args) -> int:
        """Memory management"""
        if args.memory_action == 'stats':
            if self.ensure_orchestrator():
                status = self.orchestrator.get_system_status()
                print("\n🧠 Memory Statistics:")
                mem = status.memory_usage
                print(f"   Data files: {mem.get('data_files', 0)}")
                print(f"   Total size: {mem.get('total_size_mb', 0):.1f} MB")
                return 0
            return 1
        else:
            self.print_status(f"Memory action '{args.memory_action}' not implemented", "warning")
            return 0
    
    def run(self, args=None) -> int:
        """Run the CLI with given arguments"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Set global options
        self.data_dir = parsed_args.data_dir
        self.verbose = parsed_args.verbose
        
        # Ensure data directory exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        if not parsed_args.command:
            parser.print_help()
            return 0
        
        # Route to appropriate command handler
        command_map = {
            'start': self.cmd_start,
            'status': self.cmd_status,
            'stop': self.cmd_stop,
            'migrate': self.cmd_migrate,
            'integrate': self.cmd_integrate,
            'analyze': self.cmd_analyze,
            'health': self.cmd_health,
            'backup': self.cmd_backup,
            'cleanup': self.cmd_cleanup,
            'interact': self.cmd_interact,
            'learn': self.cmd_learn,
            'chat': self.cmd_chat,
            'config': self.cmd_config,
            'memory': self.cmd_memory,
        }
        
        handler = command_map.get(parsed_args.command)
        if handler:
            return handler(parsed_args)
        else:
            self.print_status(f"Unknown command: {parsed_args.command}", "error")
            return 1

def main():
    """Main entry point"""
    cli = DualBrainCLI()
    return cli.run()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)