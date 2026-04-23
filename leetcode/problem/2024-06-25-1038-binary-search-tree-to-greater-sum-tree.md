---
layout: leetcode-entry
title: "1038. Binary Search Tree to Greater Sum Tree"
permalink: "/leetcode/problem/2024-06-25-1038-binary-search-tree-to-greater-sum-tree/"
leetcode_ui: true
entry_slug: "2024-06-25-1038-binary-search-tree-to-greater-sum-tree"
---

[1038. Binary Search Tree to Greater Sum Tree](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/description/) medium
[blog post](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/solutions/5364892/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25062024-1038-binary-search-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1z9Yicjf5bM)
![2024-06-25_07-02_1.webp](/assets/leetcode_daily_images/43e81aa0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/650

#### Problem TLDR

Aggregate Binary Search Tree from the right #medium #tree

#### Intuition

Just iterate from the tail in an inorder DFS traversal.

![2024-06-25_06-24.webp](/assets/leetcode_daily_images/b66e7615.webp)

#### Approach

* notice how `26` jumps straight to the root, so we must store the result somewhere
* there is a nice patter in Rust: `let Some(..) = .. else { .. }

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ for the call stack, however, it can be O(1) for the Morris Traversal

#### Code

```kotlin

    var s = 0
    fun bstToGst(root: TreeNode?): TreeNode? = root?.apply {
        bstToGst(right); `val` += s; s = `val`; bstToGst(left)
    }

```
```rust

    pub fn bst_to_gst(mut root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, s: i32) -> i32 {
            let Some(n) = n.as_ref() else { return s }; let mut n = n.borrow_mut();
            n.val += dfs(&n.right, s); dfs(&n.left, n.val)
        }
        dfs(&root, 0); root
    }

```

