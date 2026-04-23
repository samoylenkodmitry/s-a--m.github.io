---
layout: leetcode-entry
title: "1339. Maximum Product of Splitted Binary Tree"
permalink: "/leetcode/problem/2026-01-07-1339-maximum-product-of-splitted-binary-tree/"
leetcode_ui: true
entry_slug: "2026-01-07-1339-maximum-product-of-splitted-binary-tree"
---

[1339. Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description) medium
[blog post](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/solutions/7474724/kotlin-rust-by-samoylenkodmitry-y523/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07012026-1339-maximum-product-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/v5xmJIcPlbs)

![cc4ddd10-55f6-4511-819b-97e43c288f51 (1).webp](/assets/leetcode_daily_images/29623bf2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1230

#### Problem TLDR

Max sumA*sumB of a tree split #medium

#### Intuition

Find the total sum, then subtract each subtree to find the other sum.

#### Approach

* we can calculate in 32 bit if use extemum of x = s/2
* we can collect visited sums into a hashset
* we can skip sums that are smaller than max/2
* the fastest runtime speed is still calculation of x*(max-x) in-place

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(logn)$$

#### Code

```kotlin
// 11ms
    fun maxProduct(r: TreeNode?): Long {
        val a = ArrayList<Int>()
        fun s(r: TreeNode?): Int = r
            ?.run {val s = s(left)+s(right)+`val`; a += s; s}?:0
        val s = s(r); return a.maxOf {1L*it*(s-it)}%1000000007
    }
```
```rust
// 3ms
    pub fn max_product(r: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn s(r: &Option<Rc<RefCell<TreeNode>>>, res: &mut i64, max: &mut i32) -> i32 {
            let Some(n) = r else { return 0 }; let n = n.borrow();
            let s = s(&n.left,res,max)+s(&n.right,res,max)+n.val; *max = s.max(*max);
            *res = (*res).max(s as i64*(*max - s)as i64); s
        }
        let (mut res,mut max) = (0,0); for _ in 0..2 {s(&r, &mut res, &mut max);} (res%1000000007) as i32
    }
```

