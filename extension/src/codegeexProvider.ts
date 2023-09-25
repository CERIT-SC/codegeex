import { CodeEditor } from '@jupyterlab/codeeditor';

import { CompletionHandler } from '@jupyterlab/completer';
import { ICompletionContext, ICompletionProvider } from '@jupyterlab/completer';

export const CONTEXT_PROVIDER_ID = 'CompletionProvider:context';

export class CodegeexCompleterProvider implements ICompletionProvider {
  readonly identifier = CONTEXT_PROVIDER_ID;

  readonly rank: number = 10000;

  readonly renderer = null;

  async isApplicable(context: ICompletionContext): Promise<boolean> {
    return true;
  }

  fetch(
    request: CompletionHandler.IRequest,
    context: ICompletionContext
  ): Promise<CompletionHandler.ICompletionItemsReply> {
    const editor = context.editor;
    if (!editor) {
      return Promise.reject('No editor');
    }
    return Private.codegeexHint(editor!);
  }
}

namespace Private {
  export async function codegeexHint(
    editor: CodeEditor.IEditor
  ): Promise<CompletionHandler.ICompletionItemsReply> {
    const token = editor.getTokenAtCursor();

    const position = editor.getCursorPosition();
    const source = editor.model.sharedModel.source;

    const linesBeforeCursor = source.split('\n').slice(0, position.line + 1);
    linesBeforeCursor[position.line] = linesBeforeCursor[position.line].slice(
      0,
      position.column
    );

    const textBeforeCursor = linesBeforeCursor.join('\n');

    let codeList: string[] = [];

    const result = await fetch(
      'https://codegeex.dyn.cloud.e-infra.cz/multilingual_code_generate_adapt',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt: textBeforeCursor,
          lang: 'python',
          // temperature: 0.2,
          top_p: 0.9,
          top_k: 3,
          max_length: 200,
          n: numberOfResults(textBeforeCursor.length)
        })
      }
    );
    const data = await result.json();

    codeList = data.result.output.code;

    const matches = new Set<string>(codeList);
    const items = new Array<CompletionHandler.ICompletionItem>();
    matches.forEach(label => items.push({ label }));

    return {
      start: token.offset + token.value.length,
      end: token.offset + token.value.length,
      items
    };
  }

  const numberOfResults = (contextLength: number) => {
    if (contextLength < 100) {
      return 3;
    } else if (contextLength < 250) {
      return 2;
    } else {
      return 1;
    }
  };
}
