---
layout: leetcode-entry
title: "2471. Minimum Number of Operations to Sort a Binary Tree by Level"
permalink: "/leetcode/problem/2024-12-23-2471-minimum-number-of-operations-to-sort-a-binary-tree-by-level/"
leetcode_ui: true
entry_slug: "2024-12-23-2471-minimum-number-of-operations-to-sort-a-binary-tree-by-level"
---

[2471. Minimum Number of Operations to Sort a Binary Tree by Level](https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/solutions/6176931/kotlin-rust-by-samoylenkodmitry-i8b5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23122024-2471-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/I_UvQOt73Kw)
[deep-dive](https://notebooklm.google.com/notebook/9e4ea372-b535-416d-a981-7d1222df6bdb/audio)
![1.webp](/assets/leetcode_daily_images/25356315.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/841

#### Problem TLDR

Min swaps to sort tree layers #medium #cycle-sort #bfs

#### Intuition

Can't solve without a hint.
The hint: cycle-sort has optimal swaps count.

```j

    // 0 1 2 3 4 5
    // 4 5 1 0 3 2
    // 0     4
    //   1 5
    //     2     5
    //       3 4

    //
    // 7 6 5 4
    // 0 1 2 3
    // 3 2 1 0

```

To do the cycle-sort, we convert the layer numbers into indices sorted accordingly, then do `swap(ix[i], ix[ix[i]])` until all indices at their places.

#### Approach

* we can use a hashmap or just an array for indices to value mapping

#### Complexity

- Time complexity:
$$O(nlog(n))$$ to sort layers

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minimumOperations(root: TreeNode?): Int {
        val q = ArrayDeque<TreeNode>(listOf(root ?: return 0))
        var res = 0
        while (q.size > 0) {
            val s = ArrayList<Int>()
            repeat(q.size) {
                val n = q.removeFirst(); s += n.`val`
                n.left?.let { q += it }; n.right?.let { q += it }
            }
            val ix = s.indices.sortedBy { s[it] }.toIntArray()
            for (i in 0..<ix.size) while (ix[i] != i) {
                ix[i] = ix[ix[i]].also { ix[ix[i]] = ix[i] }; res++
            }
        }
        return res
    }

```
```rust

    pub fn minimum_operations(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        let Some(r) = root else { return 0 };
        let (mut q, mut r) = (VecDeque::from([r]), 0);
        while q.len() > 0 {
            let mut s = vec![];
            for _ in 0..q.len() {
                let n = q.pop_front().unwrap(); let n = n.borrow(); s.push(n.val);
                if let Some(n) = n.left.clone() { q.push_back(n) }
                if let Some(n) = n.right.clone() { q.push_back(n) }
            }
            let mut ix: Vec<_> = (0..s.len()).collect();
            ix.sort_unstable_by_key(|&i| s[i]);
            for i in 0..ix.len() { while ix[i] != i {
                let t = ix[i]; ix[i] = ix[t]; ix[t] = t; r += 1
            }}
        }; r
    }

```
```c++

    int minimumOperations(TreeNode* root) {
        int r = 0; vector<TreeNode*> q{root};
        while (q.size()) {
            vector<TreeNode*> q1; vector<int> s, ix(q.size());
            for (auto n: q) {
                s.push_back(n->val);
                if (n->left) q1.push_back(n->left);
                if (n->right) q1.push_back(n->right);
            }
            iota(begin(ix), end(ix), 0);
            sort(begin(ix), end(ix), [&](int i, int j){ return s[i] < s[j];});
            for (int i = 0; i < ix.size(); ++i) for (; ix[i] != i; ++r)
                swap(ix[i], ix[ix[i]]);
            swap(q, q1);
        } return r;
    }

```

