---
layout: leetcode-entry
title: "1382. Balance a Binary Search Tree"
permalink: "/leetcode/problem/2026-02-09-1382-balance-a-binary-search-tree/"
leetcode_ui: true
entry_slug: "2026-02-09-1382-balance-a-binary-search-tree"
---

[1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/description) medium
[blog post](https://leetcode.com/problems/balance-a-binary-search-tree/solutions/7565367/kotlin-rust-by-samoylenkodmitry-492m/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09022026-1382-balance-a-binary-search?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jBgLB7dJP0k)

![16abb998-7e00-4441-a0c6-61995998c0f3 (1).webp](/assets/leetcode_daily_images/66a5d686.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1263

#### Problem TLDR

Balance binary search tree #medium #dfs

#### Intuition

Collect to a list with in-order dfs.
Build a new, count of left subtree is equal to the count of right subtree. Mid is current.

#### Approach

* we can store nodes itself on a list
* we can avoid building the list, just make a lazy iterator (sequence in Kotlin, or Stack and from_fn in Rust)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, O(log(n)) for the lazy iterator

#### Code

```kotlin
// 22ms
    fun balanceBST(r: TreeNode?): TreeNode? {
        val l = buildList {fun c(r: TreeNode?) {r?.run { c(left); add(r); c(right) }};c(r)}
        fun b(f: Int, t: Int): TreeNode? =
            if (f > t) null else l[(f+t)/2].apply {
                left = b(f, (f+t)/2-1); right = b((f+t)/2+1, t)
            }
        return b(0, l.lastIndex)
    }
```
```rust
// 0ms
    type Tr = Rc<RefCell<TreeNode>>; type Opt = Option<Tr>;

    pub fn balance_bst(r: Opt) -> Opt {
        fn c(n: &Opt) -> i32 { n.as_ref().map_or(0, |n| 1 + c(&n.borrow().left) + c(&n.borrow().right)) }
        let (n, mut s, mut c) = (c(&r), vec![], r);
        let mut i = std::iter::from_fn(move || {
            while let Some(t) = c.take() { c = t.borrow().left.clone(); s.push(t); }
            s.pop().map(|t| { c = t.borrow().right.clone(); t })
        });
        fn b(k: i32, i: &mut impl Iterator<Item = Tr>) -> Opt {
            if k < 1 { return None }; let l = b(k / 2, i);
            i.next().map(|t| { t.borrow_mut().left = l; t.borrow_mut().right = b(k - 1 - k / 2, i); t })
        }
        b(n, &mut i)
    }
```

