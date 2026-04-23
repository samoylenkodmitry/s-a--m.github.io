---
layout: leetcode-entry
title: "513. Find Bottom Left Tree Value"
permalink: "/leetcode/problem/2024-02-28-513-find-bottom-left-tree-value/"
leetcode_ui: true
entry_slug: "2024-02-28-513-find-bottom-left-tree-value"
---

[513. Find Bottom Left Tree Value](https://leetcode.com/problems/find-bottom-left-tree-value/description/) medium
[blog post](https://leetcode.com/problems/find-bottom-left-tree-value/solutions/4793004/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28022024-513-find-bottom-left-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KIQiT0p1PYE)
![image.png](/assets/leetcode_daily_images/cc3a068c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/522

#### Problem TLDR

Leftmost node value of the last level of the Binary Tree.

#### Intuition

Just solve this problem for both `left` and `right` children, then choose the winner with most `depth`.

#### Approach

Code looks nicer when `dfs` function accepts nullable value.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun findBottomLeftValue(root: TreeNode?): Int {
    fun dfs(n: TreeNode?): List<Int> = n?.run {
      if (left == null && right == null) listOf(`val`, 1) else {
        val l = dfs(left); val r = dfs(right)
        val m = if (r[1] > l[1]) r else l
        listOf(m[0], m[1] + 1)
    }} ?: listOf(Int.MIN_VALUE, 0)
    return dfs(root)[0]
  }

```
```rust

  pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(n: &Option<Rc<RefCell<TreeNode>>>) -> (i32, i32) {
      n.as_ref().map_or((i32::MIN, 0), |n| { let n = n.borrow();
        if !n.left.is_some() && !n.right.is_some() { (n.val, 1) } else {
          let (l, r) = (dfs(&n.left), dfs(&n.right));
          let m = if r.1 > l.1 { r } else { l };
          (m.0, m.1 + 1)
      }})}
    dfs(&root).0
  }

```

