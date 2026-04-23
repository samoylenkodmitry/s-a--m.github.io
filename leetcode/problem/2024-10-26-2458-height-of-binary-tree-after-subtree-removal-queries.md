---
layout: leetcode-entry
title: "2458. Height of Binary Tree After Subtree Removal Queries"
permalink: "/leetcode/problem/2024-10-26-2458-height-of-binary-tree-after-subtree-removal-queries/"
leetcode_ui: true
entry_slug: "2024-10-26-2458-height-of-binary-tree-after-subtree-removal-queries"
---

[2458. Height of Binary Tree After Subtree Removal Queries](https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/description/) hard
[blog post](https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/solutions/5969996/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26102024-2458-height-of-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Y4Uurs3uKVE)
[deep-dive](https://notebooklm.google.com/notebook/e92984a2-e54f-40da-8815-9f04e9e40147/audio)
![1.webp](/assets/leetcode_daily_images/66e3f542.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/780

#### Problem TLDR

`n` new heights by cutting nodes in a Tree #hard #dfs

#### Intuition

After cutting, check the sibling: if it has the bigger depth, we are good, otherwise update and go up. This will take O(log(n)) for each call.

We can speed it up by tracking the `level` from the node upwards to the root.

The catch is the siblings of each level: there can be more than one of them. Check if the cutting node is the current level maximum depth, and if so, take the second maximum of the depth.

#### Approach

* can be done in a single DFS traversal
* in Rust `let m = ld[lvl]` makes a `copy`, do `&mut ld[lvl]` instead (silent bug)
* arrays are faster than HashMap (in the leetcode tests runner)

#### Complexity

- Time complexity:
$$O(n + q)$$

- Space complexity:
$$O(n + q)$$

#### Code

```kotlin

    fun treeQueries(root: TreeNode?, queries: IntArray): IntArray {
        val lToD = Array(100001) { intArrayOf(-1, -1) }; val vToLD = lToD.clone()
        fun dfs(n: TreeNode?, lvl: Int): Int = n?.run {
            val d = 1 + max(dfs(left, lvl + 1), dfs(right, lvl + 1))
            vToLD[`val`] = intArrayOf(lvl, d); val m = lToD[lvl]
            if (d > m[0]) { m[1] = m[0]; m[0] = d } else m[1] = max(m[1], d); d
        } ?: -1
        dfs(root, 0)
        return IntArray(queries.size) { i ->
            val (lvl, d) = vToLD[queries[i]]; val (d1, d2) = lToD[lvl]
            lvl + if (d < d1) d1 else d2
        }
    }

```
```rust

    pub fn tree_queries(root: Option<Rc<RefCell<TreeNode>>>, queries: Vec<i32>) -> Vec<i32> {
        type D = [(i32, i32); 100001];
        let mut ld = [(-1, -1); 100001]; let mut vld = ld.clone();
        fn dfs(ld: &mut D, vld: &mut D, n: &Option<Rc<RefCell<TreeNode>>>, lvl: i32) -> i32 {
            let Some(n) = n else { return -1 }; let mut n = n.borrow_mut();
            let d = 1 + dfs(ld, vld, &n.left, lvl + 1).max(dfs(ld, vld, &n.right, lvl + 1));
            vld[n.val as usize] = (lvl, d); let m = &mut ld[lvl as usize];
            if d > m.0 { m.1 = m.0; m.0 = d } else { m.1 = m.1.max(d) }; d
        }
        dfs(&mut ld, &mut vld, &root, 0);
        queries.iter().map(|&q| {
          let (lvl, d) = vld[q as usize]; let (d1, d2) = ld[lvl as usize];
            lvl + if d < d1 { d1 } else { d2 }}).collect()
    }

```
```c++

    vector<int> treeQueries(TreeNode* root, vector<int>& queries) {
        array<pair<int, int>, 100001> ld{}, vld = ld;
        function<int(TreeNode*,int)> f = [&](TreeNode* n, int l) {
            if (!n) return 0;
            int d = 1 + max(f(n->left, l + 1), f(n->right, l + 1));
            vld[n->val] = {l, d}; auto& [d1, d2] = ld[l];
            if (d > d1) d2 = d1, d1 = d; else d2 = max(d2, d);
            return d;
        };
        f(root,0);
        transform(begin(queries), end(queries), begin(queries), [&](int q){
            auto [l, d] = vld[q]; auto [d1, d2] = ld[l]; return l - 1 + (d < d1 ? d1 : d2);
        });
        return queries;
    }

```
