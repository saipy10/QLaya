import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import { QLAYA_MODELS, resolveQLModel, QLAYA_ONNX_FILES } from "../src/router.js";
import { feedMonolithicRow, type Batch } from "../src/providers.js";

describe("monolithic onnx and QLAYA_MODELS registry", () => {
  it("maps teacher models to root subfolder with onnxFile", () => {
    const spec = resolveQLModel("QLaya-int8");
    expect(spec.subfolder).toBeNull();
    expect(spec.onnxFile).toBe("qlaya.int8.onnx");

    const fp16 = resolveQLModel("QLaya-fp16");
    expect(fp16.subfolder).toBeNull();
    expect(fp16.onnxFile).toBe("qlaya.fp16.onnx");
  });

  it("maps distilled models to their actual student subfolders", () => {
    const spec14 = resolveQLModel("QLaya-14L-int8");
    expect(spec14.subfolder).toBe("distil_qlaya_14l");
    expect(spec14.onnxFile).toBe("distil_qlaya_14l.int8.onnx");

    const spec6 = resolveQLModel("QLaya-6L-int8");
    expect(spec6.subfolder).toBe("distil_qlaya_6l");
    expect(spec6.onnxFile).toBe("distil_qlaya_6l.int8.onnx");

    const spec6_4 = resolveQLModel("6l-int4");
    expect(spec6_4.subfolder).toBe("distil_qlaya_6l");
    expect(spec6_4.onnxFile).toBe("distil_qlaya_6l.int4.onnx");
  });

  it("all QLAYA_MODELS have defined onnxFile matching QLAYA_ONNX_FILES", () => {
    for (const [id, spec] of Object.entries(QLAYA_MODELS)) {
      expect(spec.onnxFile).toBeDefined();
      if (id in QLAYA_ONNX_FILES) {
        expect(spec.onnxFile).toBe(QLAYA_ONNX_FILES[id]);
      }
    }
  });

  it("feedMonolithicRow produces correct tensor shapes and 1D qtype", () => {
    const fakeOrt = {
      Tensor: class {
        type: string;
        data: any;
        dims: number[];
        constructor(type: string, data: any, dims: number[]) {
          this.type = type;
          this.data = data;
          this.dims = dims;
        }
      },
    };

    const b: Batch = {
      inputIds: [[101, 2054, 102], [101, 1037, 102]],
      attentionMask: [[1, 1, 1], [1, 1, 1]],
      markerPos: [[1, 2], [1, 2]],
      markerMask: [[true, true], [true, true]],
      qtype: [0, 1],
    };

    const row0 = feedMonolithicRow(fakeOrt, b, 0);
    expect(row0.input_ids.dims).toEqual([1, 3]);
    expect(row0.attention_mask.dims).toEqual([1, 3]);
    expect(row0.marker_pos.dims).toEqual([1, 2]);
    expect(row0.marker_mask.dims).toEqual([1, 2]);
    // Critical: qtype must be rank-1 [1], not [1, 1]
    expect(row0.qtype.dims).toEqual([1]);

    const row1 = feedMonolithicRow(fakeOrt, b, 1);
    expect(row1.qtype.dims).toEqual([1]);
  });

  it("Agent executes through monolithic provider run(batch) with multiple questions", async () => {
    let runCalledWithBatch: Batch | null = null;
    const fakeMonoProvider = {
      async run(b: Batch) {
        runCalledWithBatch = b;
        return {
          logits: [[2.5, 0.5], [1.0, 3.0]],
          act: [[1.0, 0.0], [0.8, 0.2]],
        };
      },
    };

    const agent = new Agent({ provider: fakeMonoProvider as any } as any);

    const questions = {
      q1: { type: "choice", instructions: "Choose?", criteria: { a: "First", b: "Second" } },
      q2: { type: "choice", instructions: "Pick?", criteria: { x: "X", y: "Y" } },
    };

    const res: any = await agent.systemOne("test state", questions as any);
    expect(runCalledWithBatch).not.toBeNull();
    expect(res.answers.q1.choice).toBe("a");
    expect(res.answers.q2.choice).toBe("y");
  });
});
