---
layout: leetcode-entry
title: "110. Balanced Binary Tree"
permalink: "/leetcode/problem/2026-02-08-110-balanced-binary-tree/"
leetcode_ui: true
entry_slug: "2026-02-08-110-balanced-binary-tree"
---

[110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/description) easy
[blog post](https://leetcode.com/problems/balanced-binary-tree/solutions/7563036/kotlin-rust-by-samoylenkodmitry-fs76/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08022026-110-balanced-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zjIc3QUg2KI)

![a2bb7e56-105b-45cc-a303-72c60366d33c (1).webp](/assets/leetcode_daily_images/b6453dc6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1262

#### Problem TLDR

Is tree balanced? #easy #dfs

#### Intuition

Solve the sub-problem for every node.
Compare max depths for the left and right.

#### Approach

* we can use 'marker' depth as a boolean
* we can shortcircuit and don't check the right subtree
* we can override values in a tree to golf the solutino
* BFS will not solve this: leafs can be at any heights, only max depths left&right for each node matters

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin
// 0ms
    fun isBalanced(r: TreeNode?): Boolean = r?.run {
        val b = isBalanced(left) && isBalanced(right)
        val l = left?.`val`?:0; val r = right?.`val`?:0
        `val`= 1 + max(l,r); b && abs(l-r) < 2
    } ?: true
```
```rust
// 0ms
    pub fn is_balanced(r: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn d(r: &Option<Rc<RefCell<TreeNode>>>) -> Result<i8, ()> {
            let Some(n) = r else { return Ok(0) }; let n = n.borrow();
            let l = d(&n.left)?; let r = d(&n.right)?;
            if (l-r).abs() > 1 { Err(()) } else { Ok(1 + l.max(r)) }
        }
        d(&r).is_ok()
    }
```

