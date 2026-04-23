---
layout: leetcode-entry
title: "1022. Sum of Root To Leaf Binary Numbers"
permalink: "/leetcode/problem/2026-02-24-1022-sum-of-root-to-leaf-binary-numbers/"
leetcode_ui: true
entry_slug: "2026-02-24-1022-sum-of-root-to-leaf-binary-numbers"
---

[1022. Sum of Root To Leaf Binary Numbers](https://open.substack.com/pub/dmitriisamoilenko/p/24022026-1022-sum-of-root-to-leaf?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/24022026-1022-sum-of-root-to-leaf?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24022026-1022-sum-of-root-to-leaf?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/s0FsciC8tGY)

![ff6f701b-2549-4e1f-94d5-5e39d794d8cd (1).webp](/assets/leetcode_daily_images/b796c5ce.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1279

#### Problem TLDR

Sum of a binary numbers in a binary tree #easy

#### Intuition

The simplest way: helper method, global sum variable, track leafs.

#### Approach

* we can skip checking the leafs
* we can use tree itself as a storage

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin
// 18ms
    fun sumRootToLeaf(r: TreeNode?): Int = r?.run {
        max(`val`, setOf(left,right).sumOf { it?.`val` += `val`*2; sumRootToLeaf(it) })
    } ?: 0
```
```rust
// 0ms
    pub fn sum_root_to_leaf(r: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        r.map_or(0, |n| { let n = n.borrow_mut();
            [&n.left, &n.right].into_iter().flatten().map(|x| {
                x.borrow_mut().val += n.val * 2;
                Self::sum_root_to_leaf(Some(x.clone()))}).sum::<i32>().max(n.val)
        })
    }
```

