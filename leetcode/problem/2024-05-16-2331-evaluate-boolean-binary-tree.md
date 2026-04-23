---
layout: leetcode-entry
title: "2331. Evaluate Boolean Binary Tree"
permalink: "/leetcode/problem/2024-05-16-2331-evaluate-boolean-binary-tree/"
leetcode_ui: true
entry_slug: "2024-05-16-2331-evaluate-boolean-binary-tree"
---

[2331. Evaluate Boolean Binary Tree](https://leetcode.com/problems/evaluate-boolean-binary-tree/description/) easy
[blog post](https://leetcode.com/problems/evaluate-boolean-binary-tree/solutions/5163912/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16052024-2331-evaluate-boolean-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/dyQ47TG5fpc)
![2024-05-16_08-48.webp](/assets/leetcode_daily_images/5544f403.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/604

#### Problem TLDR

Evaluate tree where `0/1` is `false/true` and `2/3` is `or/and` #easy #tree #dfs

#### Intuition

We can solve a subproblem for each node in a recursion.

#### Approach

Let's try to avoid the double walk by changing the boolean operations order.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$ for the recursion depth

#### Code

```kotlin

    fun evaluateTree(root: TreeNode?): Boolean = root?.run {
    if (`val` < 1) false else `val` < 2
    || evaluateTree(left) && (`val` < 3 || evaluateTree(right))
    || `val` < 3 && evaluateTree(right) } ?: false

```
```rust

    pub fn evaluate_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        root.as_ref().map_or(false, |n| { let mut n = n.borrow_mut();
            if n.val < 1 { false } else {
            n.val < 2 || Self::evaluate_tree(n.left.take())
            && (n.val < 3 || Self::evaluate_tree(n.right.take()))
            || n.val < 3 && Self::evaluate_tree(n.right.take())
        }})
    }

```

