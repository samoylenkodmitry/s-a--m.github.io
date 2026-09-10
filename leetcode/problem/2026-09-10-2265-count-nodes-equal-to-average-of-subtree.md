---
layout: leetcode-entry
title: "2265. Count Nodes Equal to Average of Subtree"
permalink: "/leetcode/problem/2026-09-10-2265-count-nodes-equal-to-average-of-subtree/"
leetcode_ui: true
entry_slug: "2026-09-10-2265-count-nodes-equal-to-average-of-subtree"
---

[2265. Count Nodes Equal to Average of Subtree](https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/solutions/8513376/kotlin-rust-by-samoylenkodmitry-qqsu/) medium
[substack](https://dmitriisamoilenko.substack.com/p/10092026-2265-count-nodes-equal-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NRnDG0VWBEo)

https://dmitrysamoylenko.com/leetcode/

![10.09.2026.webp](/assets/leetcode_daily_images/10.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1478

#### Problem TLDR

Values equal subtree average

#### Intuition

Solve for subtree: count, sum, result

#### Approach

* pack two values into a single 32-bit int
* reuse the function by storing data in the tree itself

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun averageOfSubtree(r: TreeNode?): Int = r?.run {
        val v = `val`
        val res = averageOfSubtree(left) + averageOfSubtree(right)
        `val` = (left?.`val` ?: 0) + (right?.`val` ?: 0) + 1 + v * 1024
        res + if (v == `val` / 1024 / (`val` % 1024)) 1 else 0
    } ?: 0
```
```rust
    pub fn average_of_subtree(r: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        let Some(n) = r else {return 0}; let mut b = n.borrow_mut(); let v = b.val;
        let [l, r] = [&b.left, &b.right].map(|c|
            (Self::average_of_subtree(c.clone()), c.as_ref().map_or(0, |x| x.borrow().val)));
        let (c, s) = (l.1 % 1024 + r.1 % 1024 + 1, l.1 / 1024 + r.1 / 1024 + v);
        b.val = s * 1024 + c; l.0 + r.0 + (v == s / c) as i32
    }
```

