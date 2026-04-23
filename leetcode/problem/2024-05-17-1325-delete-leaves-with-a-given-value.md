---
layout: leetcode-entry
title: "1325. Delete Leaves With a Given Value"
permalink: "/leetcode/problem/2024-05-17-1325-delete-leaves-with-a-given-value/"
leetcode_ui: true
entry_slug: "2024-05-17-1325-delete-leaves-with-a-given-value"
---

[1325. Delete Leaves With a Given Value](https://leetcode.com/problems/delete-leaves-with-a-given-value/description/) easy
[blog post](https://leetcode.com/problems/delete-leaves-with-a-given-value/solutions/5168887/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17052024-1325-delete-leaves-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/fsVxCGv-UW0)
![2024-05-17_08-57.webp](/assets/leetcode_daily_images/719094ee.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/605

#### Problem TLDR

Recursively remove `target` leafs from the tree #easy #dfs #tree

#### Intuition

When dealing with Binary Trees try to solve the subproblem recursively.

#### Approach

* Notice how `drop` is used in Rust, without it borrow checker would not allow to return `Some(node)`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ for the recursion depth

#### Code

```kotlin

    fun removeLeafNodes(root: TreeNode?, target: Int): TreeNode? = root?.run {
        left = removeLeafNodes(left, target)
        right = removeLeafNodes(right, target)
        if (left == null && right == null && `val` == target) null else root
    }

```
```rust

    pub fn remove_leaf_nodes(root: Option<Rc<RefCell<TreeNode>>>, target: i32) -> Option<Rc<RefCell<TreeNode>>> {
        let node = root?; let mut n = node.borrow_mut();
        n.left = Self::remove_leaf_nodes(n.left.take(), target);
        n.right = Self::remove_leaf_nodes(n.right.take(), target);
        if n.left.is_none() && n.right.is_none() && n.val == target { None } else { drop(n); Some(node) }
    }

```

