---
layout: leetcode-entry
title: "515. Find Largest Value in Each Tree Row"
permalink: "/leetcode/problem/2024-12-25-515-find-largest-value-in-each-tree-row/"
leetcode_ui: true
entry_slug: "2024-12-25-515-find-largest-value-in-each-tree-row"
---

[515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/) medium
[blog post](https://leetcode.com/problems/find-largest-value-in-each-tree-row/solutions/6184465/kotlin-rust-by-samoylenkodmitry-p92i/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24122024-515-find-largest-value-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/v49GRhgdxgI)
[deep-dive](https://notebooklm.google.com/notebook/f539d825-6b74-4bb5-9b29-75cf70f5dbbf/audio)
![1.webp](/assets/leetcode_daily_images/d90f1123.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/843

#### Problem TLDR

Tree layers maxes #medium

#### Intuition

Do DFS or BFS

#### Approach

* lambdas in c++ are interesting, don't forget to add `&`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

    fun largestValues(root: TreeNode?): List<Int> = buildList {
        fun dfs(n: TreeNode?, d: Int): Unit = n?.run {
            if (d < size) set(d, max(get(d), `val`)) else add(`val`)
            dfs(left, d + 1); dfs(right, d + 1)
        } ?: Unit
        dfs(root, 0)
    }

```
```rust

    pub fn largest_values(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![]; let Some(r) = root.clone() else { return res };
        let mut q = VecDeque::from([r]);
        while q.len() > 0 {
            let mut max = i32::MIN;
            for _ in 0..q.len() {
                let n = q.pop_front().unwrap(); let n = n.borrow();
                max = max.max(n.val);
                if let Some(x) = n.left.clone() { q.push_back(x); }
                if let Some(x) = n.right.clone() { q.push_back(x); }
            }
            res.push(max);
        }; res
    }

```
```c++

    vector<int> largestValues(TreeNode* root) {
        vector<int> r;
        auto f = [&](this auto const& f, TreeNode* n, int d) -> void {
            if (d < r.size()) r[d] = max(r[d], n->val); else r.push_back(n->val);
            if (n->left) f(n->left, d + 1); if (n->right) f(n->right, d + 1);
        }; if (root) f(root, 0); return r;
    }

```

