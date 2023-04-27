import { LogService, MatrixClient, UserID, RoomEvent } from "matrix-bot-sdk";
import { CHATGPT_CONTEXT, CHATGPT_TIMEOUT, CHATGPT_IGNORE_MEDIA, MATRIX_DEFAULT_PREFIX_REPLY, MATRIX_DEFAULT_PREFIX, MATRIX_BLACKLIST, MATRIX_WHITELIST, MATRIX_RICH_TEXT, MATRIX_PREFIX_DM, MATRIX_THREADS, MATRIX_ROOM_BLACKLIST, MATRIX_ROOM_WHITELIST, CHATGPT_PROMPT_PREFIX, PANGEA_CLASSIFIER_PROMPT } from "./env.js";
import { RelatesTo, MessageEvent} from "./interfaces.js";
import { sendError, sendReply } from "./utils.js";

import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanChatMessage, SystemChatMessage, AIChatMessage, BaseChatMessage } from "langchain/schema";
import { OpenAI } from "langchain/llms/openai";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { LLMChain } from "langchain/chains";

import { loadQARefineChain } from "langchain/chains";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { Document } from "langchain/dist/document.js";

export default class CommandHandler {

  // Variables so we can cache the bot's display name and ID for command matching later.
  private displayName: string;
  private userId: string;
  private localpart: string;
  
  private davinci: OpenAI = new OpenAI({ temperature: 0 , modelName:"text-davinci-003", maxTokens:2});
  private loader:DirectoryLoader = new DirectoryLoader("docs/",{".txt": (path) => new TextLoader(path)});
  private docs:Document[];
  private embeddings:OpenAIEmbeddings = new OpenAIEmbeddings();
  private store:MemoryVectorStore;

  constructor(private client: MatrixClient, private langchain: ChatOpenAI) { }

  public async start() {
    await this.prepareProfile();  // Populate the variables above (async)
    this.client.on("room.message", this.onMessage.bind(this)); // Set up the event handler
  }

  private async prepareProfile() {
    this.docs = await this.loader.load();
    this.store = await MemoryVectorStore.fromDocuments(this.docs, this.embeddings);
    this.userId = await this.client.getUserId();
    this.localpart = new UserID(this.userId).localpart;
    try {
      const profile = await this.client.getUserProfile(this.userId);
      if (profile && profile['displayname']) this.displayName = profile['displayname'];
    } catch (e) {
      LogService.warn("CommandHandler", e); // Non-fatal error - we'll just log it and move on.
    }
  }

  private shouldIgnore(event: MessageEvent, roomId: string): boolean {
    if (event.sender === this.userId) return true;                                                              // Ignore ourselves
    if (MATRIX_BLACKLIST &&  MATRIX_BLACKLIST.split(" ").find(b => event.sender.endsWith(b))) return true;      // Ignore if on blacklist if set
    if (MATRIX_WHITELIST && !MATRIX_WHITELIST.split(" ").find(w => event.sender.endsWith(w))) return true;      // Ignore if not on whitelist if set
    if (MATRIX_ROOM_BLACKLIST &&  MATRIX_ROOM_BLACKLIST.split(" ").find(b => roomId.endsWith(b))) return true;  // Ignore if on room blacklist if set
    if (MATRIX_ROOM_WHITELIST && !MATRIX_ROOM_WHITELIST.split(" ").find(w => roomId.endsWith(w))) return true;  // Ignore if not on room whitelist if set
    if (Date.now() - event.origin_server_ts > 50000) return true;                                               // Ignore old messages
    if (event.content["m.relates_to"]?.["rel_type"] === "m.replace") return true;                               // Ignore edits
    if (CHATGPT_IGNORE_MEDIA && event.content.msgtype !== "m.text") return true;                                // Ignore everything which is not text if set
    return false;
  }

  private getRootEventId(event: MessageEvent): string {
    const relatesTo: RelatesTo | undefined = event.content["m.relates_to"];
    const isReplyOrThread: boolean = (relatesTo === undefined)
    return (!isReplyOrThread && relatesTo.event_id !== undefined) ? relatesTo.event_id : event.event_id;
  }

  private createLangchainArray(roomEvent:RoomEvent) {
    const content:string = roomEvent.content ? String(roomEvent.content['body']) : "";
    switch (roomEvent.sender){
      case this.userId:
        return new AIChatMessage(content);
      default:
        return new HumanChatMessage(content);
    }
  }

  /**
   * Run when `message` room event is received. The bot only sends a message if needed.
   * @returns Room event handler, which itself returns nothing
   */
  private async onMessage(roomId: string, event: MessageEvent) {
    try {
      if (this.shouldIgnore(event, roomId)) {
        console.debug("Ignoring!");
        return;
      }

      await Promise.all([
        this.client.sendReadReceipt(roomId, event.event_id),
        this.client.setTyping(roomId, true, CHATGPT_TIMEOUT)
      ]);

      // const bool_template = PANGEA_CLASSIFIER_PROMPT;
      // const bool_prompt = new PromptTemplate({template: bool_template, inputVariables: ["message"] });
      // const bool_chain = new LLMChain({ llm: this.davinci, prompt:bool_prompt });
      // const bool_res = await bool_chain.call({ message: event.content.body });
      // const about_pangea = (bool_res.text.trim().toLowerCase() === 'true');

      let ai_response:string;

      // if (about_pangea) {
      //   // Run the QA Chain through the available documents
      //   const relevantDocs = await this.store.similaritySearch(event.content.body);
      //   const chain = loadQARefineChain(this.langchain);
      //   const response = await chain.call({input_documents: relevantDocs, question:event.content.body});
      //   ai_response = response.output_text;
      // }else{
      const roomContext = await this.client.getEventContext(roomId, event.event_id, 100);
      const messages = roomContext.before.filter(e => ((e.type == 'm.room.message') && (Date.now() - e.timestamp < 5400000)));
      const langchainArray = messages.map(this.createLangchainArray, this);

      langchainArray.push(new SystemChatMessage(CHATGPT_PROMPT_PREFIX))
      langchainArray.reverse()
      langchainArray.push(new HumanChatMessage(event.content.body))
      const response = await this.langchain.call(langchainArray)
      ai_response = response.text
      //}

      await Promise.all([
        this.client.setTyping(roomId, false, 500),
        sendReply(this.client, roomId, this.getRootEventId(event), `${ai_response}`, MATRIX_THREADS, MATRIX_RICH_TEXT)
      ]);

    } catch (err) {
      console.error(err);
    }
  }
}
