import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  CompletionProviderManager,
  ICompletionProviderManager
} from '@jupyterlab/completer';
import { CodegeexCompleterProvider } from './codegeexProvider';

/**
 * Initialization data for the jupyter-codegeex extension.
 */
const defaultProvider: JupyterFrontEndPlugin<void> = {
  id: '@jupyterlab/codegeex-completer-extension',
  description: 'Codegeex completer extension',
  requires: [ICompletionProviderManager],
  autoStart: true,
  activate: (
    app: JupyterFrontEnd,
    completionManager: ICompletionProviderManager
  ): void => {
    completionManager.registerProvider(new CodegeexCompleterProvider());
  }
};

const manager: JupyterFrontEndPlugin<ICompletionProviderManager> = {
  id: 'codegeex-completer-manager:plugin',
  description: 'Codegeex code completion manager',
  autoStart: true,
  requires: [],
  provides: ICompletionProviderManager,
  activate: (app: JupyterFrontEnd) => {
    const manager = new CompletionProviderManager();
    const updateSetting = (): void => {
      manager.setTimeout(10000);
      manager.setShowDocumentationPanel(true);
      manager.setContinuousHinting(false);
      manager.registerProvider(new CodegeexCompleterProvider());

      const providers = manager.getProviders();
      const sortedProviders = Object.entries(providers ?? {})
        .sort(([, rank1], [, rank2]) => rank2 - rank1)
        .map(item => item[0]);
      manager.activateProvider(sortedProviders);
    };

    updateSetting();

    return manager;
  }
};

const plugins: JupyterFrontEndPlugin<any>[] = [manager, defaultProvider];
export default plugins;
