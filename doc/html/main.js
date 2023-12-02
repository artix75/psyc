window.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('.identifier-ref').forEach(elem => {
        let ident_id = elem.getAttribute('data-identifier-id');
        /*console.log(ident_id);*/
        if (!ident_id) return;
        let parent_id = elem.getAttribute('data-parent-id');
        if (!parent_id) return;
        let sel = `#${parent_id} code *[data-identifier-id=${ident_id}]`;
        let target = document.querySelector(sel);
        if (target) {
            elem.addEventListener('mouseover', (event) => {
                target.classList.add('highlight');
            });
            elem.addEventListener('mouseout', (event) => {
                target.classList.remove('highlight');
            });
        }
    });
    document.querySelectorAll('.details .tag').forEach(elem => {
        let href = elem.getAttribute('data-href');
        elem.addEventListener('click', (event) => {
            location.href = href;
        });
    });
});
