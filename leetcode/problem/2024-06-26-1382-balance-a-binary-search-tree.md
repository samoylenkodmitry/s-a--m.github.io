---
layout: leetcode-entry
title: "1382. Balance a Binary Search Tree"
permalink: "/leetcode/problem/2024-06-26-1382-balance-a-binary-search-tree/"
leetcode_ui: true
entry_slug: "2024-06-26-1382-balance-a-binary-search-tree"
---

[1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/description/) medium
[blog post](https://leetcode.com/problems/balance-a-binary-search-tree/solutions/5370166/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26062024-1382-balance-a-binary-search?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mg8PhviL2_k)
![2024-06-26_06-42_1.webp](/assets/leetcode_daily_images/bd9d9525.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/651

#### Problem TLDR

Make a balanced Binary Search Tree #medium

#### Intuition

Construct it back from a sorted array: always peek the middle.

#### Approach

* notice how slices in Rust are helping to reduce the complexity

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun balanceBST(root: TreeNode?): TreeNode? {
        val sorted = mutableListOf<Int>()
        fun dfs1(n: TreeNode?): Unit? = n?.run {
            dfs1(left); sorted += `val`; dfs1(right) }
        fun dfs2(lo: Int, hi: Int): TreeNode? =
            if (lo > hi) null else {
            val mid = (lo + hi) / 2
            TreeNode(sorted[mid]).apply {
                left = dfs2(lo, mid - 1); right = dfs2(mid + 1, hi)
            }}
        dfs1(root); return dfs2(0, sorted.lastIndex)
    }

```
```rust

    pub fn balance_bst(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn dfs1(n: &Option<Rc<RefCell<TreeNode>>>, sorted: &mut Vec<i32>) {
            let Some(n) = n.as_ref() else { return; }; let n = n.borrow();
            dfs1(&n.left, sorted); sorted.push(n.val); dfs1(&n.right, sorted)
        }
        fn dfs2(sorted: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
            if sorted.len() < 1 { return None }; let mid = sorted.len() / 2;
            let left = dfs2(&sorted[..mid]);
            let right = dfs2(&sorted[mid + 1..]);
            Some(Rc::new(RefCell::new(TreeNode { val: sorted[mid], left: left, right: right })))
        }
        let mut sorted = vec![]; dfs1(&root, &mut sorted); dfs2(&sorted[..])
    }

```

